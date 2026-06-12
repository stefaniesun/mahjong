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
    cookie_path = Path(args.cookies).expanduser() if args.cookies else None
    if cookie_path:
        if not cookie_path.exists():
            raise FetchVideosError(f"Cookie 文件不存在: {cookie_path}")
        return ["--cookies", str(cookie_path)]
    if args.browser:
        return ["--cookies-from-browser", args.browser]
    return []


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


def list_bili_videos(source: SourceEntry, args: argparse.Namespace) -> list[VideoCandidate]:
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "--dump-single-json",
        source.url,
    ]
    command.extend(ensure_cookie_args(args))
    completed = run_command(command)
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
    return [path for path in output_dir.glob(f"{candidate.id}.*") if path.suffix.lower() not in {".json", ".part", ".ytdl"}]


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


def process_source(source: SourceEntry, args: argparse.Namespace, state: dict[str, Any]) -> AuthorReport:




    output_dir = Path(args.output_root).expanduser() / source.output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    report = AuthorReport(source=source.source_key, platform=source.platform, output_dir=str(output_dir))
    downloaded_ids = set(state.get("downloaded", {}).get(source.source_key, {}).keys())

    if source.platform == BILI_DIR_PREFIX:
        candidates = list_bili_videos(source, args)
    elif source.platform == DOUYIN_DIR_PREFIX:
        candidates = list_douyin_videos(source, args)
    else:
        raise FetchVideosError(f"不支持的平台: {source.platform}")

    report.discovered = len(candidates)
    max_videos = args.max_videos_override or source.max_videos or DEFAULT_MAX_VIDEOS
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
        reports: list[dict[str, Any]] = []
        for source in sources:
            report = process_source(source, args, state)
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

