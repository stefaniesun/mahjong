"""fetch_videos.py

按 `configs/sources.yaml` 批量抓取 B 站/抖音博主视频，支持关键词过滤、增量下载、礼貌限速与抓取报告。

示例：
    python scripts/fetch_videos.py --sources configs/sources.yaml --output-root data/raw_videos
    python scripts/fetch_videos.py --platform bili --limit-authors 1 --browser chrome --dry-run
    python scripts/fetch_videos.py --platform bili --cookies configs/cookies/bilibili.txt --state data/raw_videos/download_state.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_MAX_VIDEOS = 20
DEFAULT_SLEEP_MIN = 3.0
DEFAULT_SLEEP_MAX = 8.0
DEFAULT_RETRIES = 2
DEFAULT_LIST_RETRIES = 4
DEFAULT_AUTHOR_GAP_MIN = 10.0
DEFAULT_AUTHOR_GAP_MAX = 20.0
DEFAULT_LIST_BACKOFF_BASE = 30.0
DEFAULT_LIST_BACKOFF_MAX = 300.0
DEFAULT_LIST_LIMIT = 40
BILI_DIR_PREFIX = "bili"
DOUYIN_DIR_PREFIX = "dy"
SUPPORTED_PLATFORMS = {BILI_DIR_PREFIX, DOUYIN_DIR_PREFIX}


class FetchVideosError(RuntimeError):
    """业务错误，向 CLI 返回可读提示。"""


@dataclass
class SourceEntry:
    """标准化后的数据源配置。"""

    platform: str
    uid: str
    url: str
    name: str
    max_videos: int = DEFAULT_MAX_VIDEOS
    include_keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)

    @property
    def source_key(self) -> str:
        return f"{self.platform}_{self.uid}"

    @property
    def output_dirname(self) -> str:
        return self.source_key


@dataclass
class VideoCandidate:
    """候选视频元数据。"""

    id: str
    title: str
    url: str
    webpage_url: str
    uploader: str
    upload_date: str | None
    extractor: str
    raw: dict[str, Any]


@dataclass
class AuthorReport:
    """单博主抓取报告。"""

    source: str
    platform: str
    output_dir: str
    discovered: int = 0
    matched: int = 0
    downloaded: int = 0
    skipped_existing: int = 0
    skipped_keyword: int = 0
    skipped_state: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)
    dry_run_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "platform": self.platform,
            "output_dir": self.output_dir,
            "discovered": self.discovered,
            "matched": self.matched,
            "downloaded": self.downloaded,
            "skipped_existing": self.skipped_existing,
            "skipped_keyword": self.skipped_keyword,
            "skipped_state": self.skipped_state,
            "failed": self.failed,
            "failures": self.failures,
            "dry_run_candidates": self.dry_run_candidates,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量抓取 B 站/抖音博主视频并维护增量状态")
    parser.add_argument("--sources", default="configs/sources.yaml", help="博主配置 YAML 路径")
    parser.add_argument("--output-root", default="data/raw_videos", help="视频输出根目录")
    parser.add_argument("--state", default="data/raw_videos/download_state.json", help="下载状态 JSON 路径")
    parser.add_argument("--report", default="data/raw_videos/fetch_report.json", help="抓取报告 JSON 路径")
    parser.add_argument("--cookies", default="", help="Netscape 格式 Cookie 文件路径，优先级高于浏览器直读")
    parser.add_argument("--proxy", default=None, help="传给 yt-dlp 的代理地址；传空字符串可禁用环境代理")
    parser.add_argument(
        "--browser",
        choices=["chrome", "edge", "firefox", ""],
        default="chrome",
        help="未提供 --cookies 时，yt-dlp 尝试从浏览器读取 Cookie",
    )
    parser.add_argument("--platform", choices=sorted(SUPPORTED_PLATFORMS), default="", help="仅抓取指定平台")
    parser.add_argument("--limit-authors", type=int, default=0, help="仅处理前 N 个博主，便于联调")
    parser.add_argument("--max-videos-override", type=int, default=0, help="覆盖 sources.yaml 中的 max_videos")
    parser.add_argument("--sleep-min", type=float, default=DEFAULT_SLEEP_MIN, help="请求间隔最小秒数")
    parser.add_argument("--sleep-max", type=float, default=DEFAULT_SLEEP_MAX, help="请求间隔最大秒数")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="单视频失败后的重试次数")
    parser.add_argument(
        "--list-retries",
        type=int,
        default=DEFAULT_LIST_RETRIES,
        help="空间列表请求失败的重试次数，主要用于应对 B 站 412 限流",
    )
    parser.add_argument(
        "--author-gap-min",
        type=float,
        default=DEFAULT_AUTHOR_GAP_MIN,
        help="不同博主之间的最小间隔秒数（礼貌限速，规避 B 站 412）",
    )
    parser.add_argument(
        "--author-gap-max",
        type=float,
        default=DEFAULT_AUTHOR_GAP_MAX,
        help="不同博主之间的最大间隔秒数",
    )
    parser.add_argument(
        "--list-backoff-base",
        type=float,
        default=DEFAULT_LIST_BACKOFF_BASE,
        help="列表请求遭遇 412 后的首次退避秒数，按 2 的幂递增（30→60→120…）",
    )
    parser.add_argument(
        "--list-backoff-max",
        type=float,
        default=DEFAULT_LIST_BACKOFF_MAX,
        help="列表请求 412 退避的单次上限秒数",
    )
    parser.add_argument(
        "--candidates-cache",
        default="data/raw_videos/_candidates_cache.json",
        help="博主视频列表缓存路径；成功拉取一次即持久化，重跑直接复用以规避 412",
    )
    parser.add_argument(
        "--refresh-list",
        action="store_true",
        help="忽略列表缓存与配额，强制重新拉取每个博主的视频列表",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=DEFAULT_LIST_LIMIT,
        help="拉取空间列表时只取最新 N 个视频（避免翻遍整个空间触发 B 站 412）",
    )
    parser.add_argument("--download-archive", default="", help="可选：传给 yt-dlp 的下载归档文件")
    parser.add_argument(
        "--douyin-manifest",
        default="",
        help="抖音降级模式的 URL 清单 JSON 路径；当专用后端不可用时，从该文件读取待处理视频列表",
    )
    parser.add_argument("--dry-run", action="store_true", help="只枚举和过滤，不实际下载")
    return parser.parse_args(argv)


def parse_douyin_manifest_sources(manifest: dict[str, list[dict[str, Any]]]) -> list[SourceEntry]:
    sources: list[SourceEntry] = []
    for source_key, entries in manifest.items():
        if not source_key.startswith(f"{DOUYIN_DIR_PREFIX}_"):
            continue
        uid = source_key[len(f"{DOUYIN_DIR_PREFIX}_") :].strip()
        if not uid:
            continue
        first_entry = entries[0] if entries else {}
        uploader = str(first_entry.get("uploader") or "").strip() if isinstance(first_entry, dict) else ""
        name = uploader or uid
        sources.append(
            SourceEntry(
                platform=DOUYIN_DIR_PREFIX,
                uid=uid,
                url=build_profile_url(DOUYIN_DIR_PREFIX, uid),
                name=name,
            )
        )
    return sources


def load_sources(path: Path, douyin_manifest: Path | None = None) -> list[SourceEntry]:
    if not path.exists():
        raise FetchVideosError(f"sources 配置不存在: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise FetchVideosError("sources.yaml 顶层必须是列表")

    sources: list[SourceEntry] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise FetchVideosError(f"第 {idx + 1} 条 source 不是对象")
        platform = str(item.get("platform", "")).strip().lower()
        uid = str(item.get("uid", "")).strip()
        url = str(item.get("url", "")).strip()
        name = str(item.get("name", uid or url or f"source_{idx + 1}")).strip()
        if platform not in SUPPORTED_PLATFORMS:
            raise FetchVideosError(f"第 {idx + 1} 条 source 的 platform 非法: {platform}")
        if not uid and not url:
            raise FetchVideosError(f"第 {idx + 1} 条 source 必须提供 uid 或 url")
        uid = uid or infer_uid_from_url(platform, url)
        if not url:
            url = build_profile_url(platform, uid)
        max_videos = int(item.get("max_videos", DEFAULT_MAX_VIDEOS) or DEFAULT_MAX_VIDEOS)
        include_keywords = normalize_keywords(item.get("include_keywords", []))
        exclude_keywords = normalize_keywords(item.get("exclude_keywords", []))
        sources.append(
            SourceEntry(
                platform=platform,
                uid=uid,
                url=url,
                name=name,
                max_videos=max_videos,
                include_keywords=include_keywords,
                exclude_keywords=exclude_keywords,
            )
        )

    if douyin_manifest is not None:
        existing_keys = {source.source_key for source in sources}
        manifest_sources = parse_douyin_manifest_sources(load_douyin_manifest(douyin_manifest))
        for source in manifest_sources:
            if source.source_key not in existing_keys:
                sources.append(source)
                existing_keys.add(source.source_key)
    return sources



def normalize_keywords(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Iterable):
        return [str(v).strip() for v in value if str(v).strip()]
    raise FetchVideosError(f"关键词字段格式非法: {value!r}")


def infer_uid_from_url(platform: str, url: str) -> str:
    cleaned = url.rstrip("/")
    if platform == BILI_DIR_PREFIX:
        if "space.bilibili.com/" in cleaned:
            return cleaned.rsplit("/", 1)[-1].split("?", 1)[0]
    if platform == DOUYIN_DIR_PREFIX:
        if "/user/" in cleaned:
            return cleaned.rsplit("/user/", 1)[-1].split("?", 1)[0]
    return cleaned.replace("https://", "").replace("http://", "").replace("/", "_")


def build_profile_url(platform: str, uid: str) -> str:
    if platform == BILI_DIR_PREFIX:
        return f"https://space.bilibili.com/{uid}"
    if platform == DOUYIN_DIR_PREFIX:
        return f"https://www.douyin.com/user/{uid}"
    raise FetchVideosError(f"不支持的平台: {platform}")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"downloaded": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FetchVideosError(f"状态文件不是合法 JSON: {path}") from exc
    if not isinstance(data, dict):
        raise FetchVideosError("状态文件顶层必须是对象")
    data.setdefault("downloaded", {})
    if not isinstance(data["downloaded"], dict):
        raise FetchVideosError("状态文件 downloaded 字段必须是对象")
    return data


def save_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_cookie_args(args: argparse.Namespace) -> list[str]:
    extra_args = []
    if args.proxy is not None:
        extra_args.extend(["--proxy", args.proxy])

    cookie_path = Path(args.cookies).expanduser() if args.cookies else None
    if cookie_path:
        if not cookie_path.exists():
            raise FetchVideosError(f"Cookie 文件不存在: {cookie_path}")
        return extra_args + ["--cookies", str(cookie_path)]
    if args.browser:
        return extra_args + ["--cookies-from-browser", args.browser]
    return extra_args


def get_cookie_path(args: argparse.Namespace) -> Path | None:
    if not args.cookies:
        return None
    cookie_path = Path(args.cookies).expanduser()
    if not cookie_path.exists():
        raise FetchVideosError(f"Cookie 文件不存在: {cookie_path}")
    return cookie_path


def build_douyin_f2_command(source: SourceEntry, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "f2",
        "douyin",
        "user",
        "-u",
        source.url,
        "--mode",
        "post",
        "--max-count",
        str(source.max_videos),
        "--json",
    ]
    cookie_path = get_cookie_path(args)
    if cookie_path:
        command.extend(["--cookie-file", str(cookie_path)])
    return command


def run_command(command: list[str], *, retries: int = 0, sleep_range: tuple[float, float] | None = None) -> subprocess.CompletedProcess[str]:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(retries + 1):
        try:
            return subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt >= retries:
                break
            if sleep_range:
                sleep_with_jitter(*sleep_range)
    assert last_error is not None
    raise last_error


def sleep_with_jitter(min_seconds: float, max_seconds: float) -> None:
    time.sleep(random.uniform(min_seconds, max_seconds))


def is_rate_limited(exc: subprocess.CalledProcessError) -> bool:
    """识别 B 站空间列表的 412 限流错误。"""
    blob = f"{exc.stderr or ''}{exc.stdout or ''}"
    return "412" in blob or "blocked by server" in blob.lower()


def run_listing_with_backoff(command: list[str], args: argparse.Namespace) -> subprocess.CompletedProcess[str]:
    """列表请求专用重试：命中 412 时按指数退避（30→60→120…）长时间等待。"""
    attempts = max(args.list_retries, 0) + 1
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(attempts):
        try:
            return run_command(command)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt >= attempts - 1:
                break
            if is_rate_limited(exc):
                wait = min(args.list_backoff_base * (2 ** attempt), args.list_backoff_max)
            else:
                wait = random.uniform(args.author_gap_min, args.author_gap_max)
            time.sleep(wait + random.uniform(0.0, 3.0))
    assert last_error is not None
    raise last_error


def candidate_to_dict(candidate: VideoCandidate) -> dict[str, Any]:
    return {
        "id": candidate.id,
        "title": candidate.title,
        "url": candidate.url,
        "webpage_url": candidate.webpage_url,
        "uploader": candidate.uploader,
        "upload_date": candidate.upload_date,
        "extractor": candidate.extractor,
    }


def candidate_from_dict(data: dict[str, Any]) -> VideoCandidate:
    return VideoCandidate(
        id=str(data.get("id", "")),
        title=str(data.get("title", "")) or str(data.get("id", "")),
        url=str(data.get("url") or data.get("webpage_url", "")),
        webpage_url=str(data.get("webpage_url") or data.get("url", "")),
        uploader=str(data.get("uploader", "")),
        upload_date=data.get("upload_date"),
        extractor=str(data.get("extractor", "")),
        raw={},
    )


def load_candidates_cache(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FetchVideosError(f"列表缓存不是合法 JSON: {path}") from exc
    return data if isinstance(data, dict) else {}


def list_videos_for_source(
    source: SourceEntry,
    args: argparse.Namespace,
    cache: dict[str, list[dict[str, Any]]],
    cache_path: Path | None,
) -> list[VideoCandidate]:
    """优先复用列表缓存；缓存未命中时拉取一次并持久化，避免重跑反复请求触发 412。"""
    if not args.refresh_list and source.source_key in cache:
        return [candidate_from_dict(item) for item in cache[source.source_key]]

    # 真正发起网络列表请求前留出较长间隔，避免连续请求空间页触发 B 站 412 限流。
    sleep_with_jitter(args.author_gap_min, args.author_gap_max)
    if source.platform == BILI_DIR_PREFIX:
        videos = list_bili_videos(source, args)
    elif source.platform == DOUYIN_DIR_PREFIX:
        videos = list_douyin_videos(source, args)
    else:
        raise FetchVideosError(f"不支持的平台: {source.platform}")

    cache[source.source_key] = [candidate_to_dict(video) for video in videos]
    if cache_path is not None:
        save_json(cache_path, cache)
    return videos


def list_bili_videos(source: SourceEntry, args: argparse.Namespace) -> list[VideoCandidate]:
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "--dump-single-json",
        # 只取最新 N 个，避免翻遍整个空间（上千视频会发起几十次分页请求直接触发 412）。
        "--playlist-end",
        str(max(args.list_limit, 1)),
        # 给少量分页请求之间留出间隔，进一步降低限流概率。
        "--sleep-requests",
        "2",
    ]
    command.extend(ensure_cookie_args(args))
    command.append(source.url)
    # 空间列表请求最易触发 B 站 412 限流，命中后按指数退避长时间等待重试。
    completed = run_listing_with_backoff(command, args)
    payload = json.loads(completed.stdout)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    videos: list[VideoCandidate] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        video_id = str(entry.get("id", "")).strip()
        title = str(entry.get("title", "")).strip() or video_id
        webpage_url = str(entry.get("url") or entry.get("webpage_url") or "").strip()
        if webpage_url and webpage_url.startswith("/"):
            webpage_url = f"https://www.bilibili.com{webpage_url}"
        if not webpage_url:
            webpage_url = f"https://www.bilibili.com/video/{video_id}"
        videos.append(
            VideoCandidate(
                id=video_id,
                title=title,
                url=webpage_url,
                webpage_url=webpage_url,
                uploader=source.name,
                upload_date=str(entry.get("upload_date") or "") or None,
                extractor=str(entry.get("extractor_key") or "BiliBili"),
                raw=entry,
            )
        )
    return videos


def load_douyin_manifest(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        raise FetchVideosError(f"抖音 URL 清单不存在: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FetchVideosError(f"抖音 URL 清单不是合法 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise FetchVideosError("抖音 URL 清单顶层必须是对象，键为 source_key")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for source_key, entries in payload.items():
        if not isinstance(entries, list):
            raise FetchVideosError(f"抖音 URL 清单中 {source_key} 的值必须是数组")
        normalized[str(source_key)] = [entry for entry in entries if isinstance(entry, dict)]
    return normalized


def build_douyin_candidate(entry: dict[str, Any], fallback_uploader: str, fallback_id: str) -> VideoCandidate | None:
    video_id = str(entry.get("aweme_id") or entry.get("id") or fallback_id).strip()
    webpage_url = str(entry.get("share_url") or entry.get("url") or entry.get("webpage_url") or "").strip()
    if not webpage_url and video_id:
        webpage_url = f"https://www.douyin.com/video/{video_id}"
    if not video_id or not webpage_url:
        return None
    title = str(entry.get("desc") or entry.get("title") or video_id).strip() or video_id
    author = entry.get("author") if isinstance(entry.get("author"), dict) else {}
    uploader = str(author.get("nickname") or entry.get("uploader") or fallback_uploader).strip() or fallback_uploader
    create_time = entry.get("create_time")
    if isinstance(create_time, (int, float)):
        upload_date = time.strftime("%Y%m%d", time.localtime(create_time))
    else:
        upload_date = str(entry.get("upload_date") or "").strip() or None
    return VideoCandidate(
        id=video_id,
        title=title,
        url=webpage_url,
        webpage_url=webpage_url,
        uploader=uploader,
        upload_date=upload_date,
        extractor="f2",
        raw=entry,
    )


def list_douyin_videos(source: SourceEntry, args: argparse.Namespace) -> list[VideoCandidate]:
    if args.douyin_manifest:
        manifest = load_douyin_manifest(Path(args.douyin_manifest).expanduser())
        entries = manifest.get(source.source_key, [])
        videos: list[VideoCandidate] = []
        for idx, entry in enumerate(entries):
            video_url = str(entry.get("url", "")).strip()
            video_id = str(entry.get("id", "")).strip() or f"{source.uid}_{idx + 1}"
            if not video_url:
                continue
            title = str(entry.get("title", "")).strip() or video_id
            videos.append(
                VideoCandidate(
                    id=video_id,
                    title=title,
                    url=video_url,
                    webpage_url=video_url,
                    uploader=str(entry.get("uploader", source.name)).strip() or source.name,
                    upload_date=str(entry.get("upload_date") or "").strip() or None,
                    extractor="manifest",
                    raw=entry,
                )
            )
        return videos

    try:
        completed = run_command(build_douyin_f2_command(source, args))
    except FileNotFoundError as exc:
        raise FetchVideosError("抖音自动抓取需要先安装 f2，或改用 --douyin-manifest 降级模式。") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        detail = stderr.splitlines()[-1] if stderr else "f2 执行失败"
        raise FetchVideosError(f"抖音自动抓取失败: {detail}；也可改用 --douyin-manifest 降级模式。") from exc

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FetchVideosError("抖音自动抓取返回的不是合法 JSON，请检查 f2 输出格式。") from exc

    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise FetchVideosError("抖音自动抓取未返回 items 列表，请检查 f2 输出格式或登录态。")

    videos: list[VideoCandidate] = []
    for idx, entry in enumerate(items):
        if not isinstance(entry, dict):
            continue
        candidate = build_douyin_candidate(entry, source.name, f"{source.uid}_{idx + 1}")
        if candidate is not None:
            videos.append(candidate)
    return videos


def filter_candidates(source: SourceEntry, candidates: list[VideoCandidate], downloaded_ids: set[str], max_videos: int) -> tuple[list[VideoCandidate], int, int]:
    matched: list[VideoCandidate] = []
    skipped_keyword = 0
    skipped_state = 0
    for candidate in candidates:
        title = candidate.title.lower()
        include_ok = not source.include_keywords or any(keyword.lower() in title for keyword in source.include_keywords)
        exclude_hit = any(keyword.lower() in title for keyword in source.exclude_keywords)
        if not include_ok or exclude_hit:
            skipped_keyword += 1
            continue
        if candidate.id in downloaded_ids:
            skipped_state += 1
            continue
        matched.append(candidate)
        if len(matched) >= max_videos:
            break
    return matched, skipped_keyword, skipped_state


def sanitize_filename(value: str) -> str:
    invalid = '<>:"/\\|?*'
    sanitized = "".join("_" if ch in invalid else ch for ch in value)
    return sanitized.strip().rstrip(".")[:120] or "video"


def info_path_for_video(output_dir: Path, candidate: VideoCandidate) -> Path:
    return output_dir / f"{candidate.id}.info.json"


def existing_video_files(output_dir: Path, candidate: VideoCandidate) -> list[Path]:
    # 下载命名为 {id}_{title}.ext，因此既要匹配 {id}.* 也要匹配 {id}_*，
    # 否则部分下载的博主在重跑时会被整体重复下载。
    skip_suffixes = {".json", ".part", ".ytdl"}
    matches: dict[str, Path] = {}
    for pattern in (f"{candidate.id}.*", f"{candidate.id}_*"):
        for path in output_dir.glob(pattern):
            if path.suffix.lower() in skip_suffixes:
                continue
            matches[path.name] = path
    return list(matches.values())


def download_bili_video(source: SourceEntry, candidate: VideoCandidate, output_dir: Path, args: argparse.Namespace) -> None:
    title_stub = sanitize_filename(candidate.title)
    template = output_dir / f"{candidate.id}_{title_stub}.%(ext)s"
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-progress",
        "--write-info-json",
        "--no-write-playlist-metafiles",
        "--output",
        str(template),
        "--format",
        "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
    ]
    command.extend(ensure_cookie_args(args))
    if args.download_archive:
        command.extend(["--download-archive", args.download_archive])
    command.append(candidate.webpage_url)
    run_command(command, retries=args.retries, sleep_range=(args.sleep_min, args.sleep_max))


def download_douyin_video(source: SourceEntry, candidate: VideoCandidate, output_dir: Path, args: argparse.Namespace) -> None:
    title_stub = sanitize_filename(candidate.title)
    template = output_dir / f"{candidate.id}_{title_stub}.%(ext)s"
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-progress",
        "--write-info-json",
        "--no-write-playlist-metafiles",
        "--output",
        str(template),
    ]
    command.extend(ensure_cookie_args(args))
    if args.download_archive:
        command.extend(["--download-archive", args.download_archive])
    command.append(candidate.webpage_url)
    run_command(command, retries=args.retries, sleep_range=(args.sleep_min, args.sleep_max))


def mark_downloaded(state: dict[str, Any], source: SourceEntry, candidate: VideoCandidate) -> None:
    source_state = state.setdefault("downloaded", {}).setdefault(source.source_key, {})
    source_state[candidate.id] = {
        "title": candidate.title,
        "webpage_url": candidate.webpage_url,
        "upload_date": candidate.upload_date,
    }


def process_source(
    source: SourceEntry,
    args: argparse.Namespace,
    state: dict[str, Any],
    cache: dict[str, list[dict[str, Any]]],
    cache_path: Path | None,
) -> AuthorReport:
    output_dir = Path(args.output_root).expanduser() / source.output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    report = AuthorReport(source=source.source_key, platform=source.platform, output_dir=str(output_dir))
    downloaded_ids = set(state.get("downloaded", {}).get(source.source_key, {}).keys())

    max_videos = args.max_videos_override or source.max_videos or DEFAULT_MAX_VIDEOS
    # 配额已满的博主直接跳过：不再请求空间列表，避免无谓地触发 B 站 412 限流。
    if not args.refresh_list and not args.dry_run and len(downloaded_ids) >= max_videos:
        report.skipped_state = len(downloaded_ids)
        return report

    candidates = list_videos_for_source(source, args, cache, cache_path)

    report.discovered = len(candidates)
    matched, skipped_keyword, skipped_state = filter_candidates(source, candidates, downloaded_ids, max_videos)
    report.matched = len(matched)
    report.skipped_keyword = skipped_keyword
    report.skipped_state = skipped_state

    for candidate in matched:
        if args.dry_run:
            report.dry_run_candidates.append(candidate.webpage_url)
            continue
        if existing_video_files(output_dir, candidate):
            report.skipped_existing += 1
            mark_downloaded(state, source, candidate)
            continue
        sleep_with_jitter(args.sleep_min, args.sleep_max)
        try:
            if source.platform == BILI_DIR_PREFIX:
                download_bili_video(source, candidate, output_dir, args)
            elif source.platform == DOUYIN_DIR_PREFIX:
                download_douyin_video(source, candidate, output_dir, args)
            mark_downloaded(state, source, candidate)
            report.downloaded += 1
        except subprocess.CalledProcessError as exc:
            report.failed += 1
            stderr = (exc.stderr or exc.stdout or "").strip().splitlines()
            reason = stderr[-1] if stderr else f"exit code {exc.returncode}"
            report.failures.append(f"{candidate.webpage_url} :: {reason}")




    return report


def select_sources(sources: list[SourceEntry], args: argparse.Namespace) -> list[SourceEntry]:

    selected = sources
    if args.platform:
        selected = [source for source in selected if source.platform == args.platform]
    if args.limit_authors > 0:
        selected = selected[: args.limit_authors]
    return selected


def main() -> int:

    args = parse_args()

    try:
        if args.sleep_min > args.sleep_max:
            raise FetchVideosError("--sleep-min 不能大于 --sleep-max")
        douyin_manifest = Path(args.douyin_manifest).expanduser() if args.douyin_manifest else None
        sources = load_sources(Path(args.sources).expanduser(), douyin_manifest=douyin_manifest)
        sources = select_sources(sources, args)
        if not sources:
            raise FetchVideosError("没有可处理的 source 条目")
        state_path = Path(args.state).expanduser()
        state = load_state(state_path)
        cache_path = Path(args.candidates_cache).expanduser() if args.candidates_cache else None
        cache = load_candidates_cache(cache_path)
        reports: list[dict[str, Any]] = []
        for source in sources:
            try:
                report = process_source(source, args, state, cache, cache_path)
            except subprocess.CalledProcessError as exc:
                output_dir = Path(args.output_root).expanduser() / source.output_dirname
                report = AuthorReport(source=source.source_key, platform=source.platform, output_dir=str(output_dir))
                report.failed = 1
                stderr = (exc.stderr or exc.stdout or "").strip().splitlines()
                reason = stderr[-1] if stderr else f"exit code {exc.returncode}"
                report.failures.append(f"{source.url} :: {reason}")
            reports.append(report.to_dict())
            if not args.dry_run:
                save_json(state_path, state)
        summary = {
            "sources": reports,
            "totals": {
                "authors": len(reports),
                "downloaded": sum(item["downloaded"] for item in reports),
                "failed": sum(item["failed"] for item in reports),
                "skipped_existing": sum(item["skipped_existing"] for item in reports),
                "skipped_keyword": sum(item["skipped_keyword"] for item in reports),
                "skipped_state": sum(item["skipped_state"] for item in reports),
            },
            "dry_run": args.dry_run,
        }
        save_json(Path(args.report).expanduser(), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except FetchVideosError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        message = stderr.splitlines()[-1] if stderr else str(exc)
        print(f"ERROR: 外部命令执行失败: {message}", file=sys.stderr)
        return exc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())

