import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import fetch_videos


class FetchVideosTests(unittest.TestCase):
    def test_filter_candidates_respects_keywords_and_state(self) -> None:
        source = fetch_videos.SourceEntry(
            platform="bili",
            uid="123",
            url="https://space.bilibili.com/123",
            name="demo",
            include_keywords=["麻将"],
            exclude_keywords=["教学"],
        )
        candidates = [
            fetch_videos.VideoCandidate(
                id="keep",
                title="四川麻将实战",
                url="u1",
                webpage_url="u1",
                uploader="demo",
                upload_date=None,
                extractor="BiliBili",
                raw={},
            ),
            fetch_videos.VideoCandidate(
                id="skip_keyword",
                title="四川麻将教学",
                url="u2",
                webpage_url="u2",
                uploader="demo",
                upload_date=None,
                extractor="BiliBili",
                raw={},
            ),
            fetch_videos.VideoCandidate(
                id="skip_state",
                title="麻将夜战",
                url="u3",
                webpage_url="u3",
                uploader="demo",
                upload_date=None,
                extractor="BiliBili",
                raw={},
            ),
        ]

        matched, skipped_keyword, skipped_state = fetch_videos.filter_candidates(
            source,
            candidates,
            downloaded_ids={"skip_state"},
            max_videos=10,
        )

        self.assertEqual([item.id for item in matched], ["keep"])
        self.assertEqual(skipped_keyword, 1)
        self.assertEqual(skipped_state, 1)

    def test_list_douyin_videos_uses_manifest_fallback(self) -> None:
        source = fetch_videos.SourceEntry(
            platform="dy",
            uid="abc",
            url="https://www.douyin.com/user/abc",
            name="douyin-author",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "douyin_urls.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dy_abc": [
                            {
                                "id": "v1",
                                "title": "四川麻将实战合集",
                                "url": "https://www.douyin.com/video/1",
                                "upload_date": "20260101",
                                "uploader": "douyin-author",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            args = fetch_videos.parse_args([
                "--douyin-manifest",
                str(manifest_path),
            ])

            videos = fetch_videos.list_douyin_videos(source, args)

        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0].id, "v1")
        self.assertEqual(videos[0].webpage_url, "https://www.douyin.com/video/1")
        self.assertEqual(videos[0].extractor, "manifest")

    def test_load_sources_infers_douyin_source_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sources_path = Path(tmpdir) / "sources.yaml"
            manifest_path = Path(tmpdir) / "douyin_urls.json"
            sources_path.write_text(
                """
- platform: bili
  uid: "123"
  url: "https://space.bilibili.com/123"
  name: "demo"
""".strip(),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "dy_abc": [
                            {
                                "id": "v1",
                                "title": "四川麻将实战合集",
                                "url": "https://www.douyin.com/video/1",
                                "uploader": "douyin-author",
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            args = fetch_videos.parse_args([
                "--sources",
                str(sources_path),
                "--douyin-manifest",
                str(manifest_path),
                "--platform",
                "dy",
            ])

            sources = fetch_videos.load_sources(Path(args.sources), douyin_manifest=manifest_path)
            filtered = fetch_videos.select_sources(sources, args)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].platform, "dy")
        self.assertEqual(filtered[0].uid, "abc")
        self.assertEqual(filtered[0].source_key, "dy_abc")
        self.assertEqual(filtered[0].name, "douyin-author")
        self.assertEqual(filtered[0].url, "https://www.douyin.com/user/abc")

    @mock.patch("scripts.fetch_videos.run_command")
    def test_list_douyin_videos_uses_f2_backend(self, mock_run_command: mock.Mock) -> None:
        mock_run_command.return_value = mock.Mock(
            stdout=json.dumps(
                {
                    "items": [
                        {
                            "aweme_id": "701",
                            "desc": "四川麻将夜战",
                            "share_url": "https://www.douyin.com/video/701",
                            "author": {"nickname": "douyin-author"},
                            "create_time": 1764547200,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )
        source = fetch_videos.SourceEntry(
            platform="dy",
            uid="abc",
            url="https://www.douyin.com/user/abc",
            name="douyin-author",
        )
        args = fetch_videos.parse_args([])

        videos = fetch_videos.list_douyin_videos(source, args)

        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0].id, "701")
        self.assertEqual(videos[0].title, "四川麻将夜战")
        self.assertEqual(videos[0].webpage_url, "https://www.douyin.com/video/701")
        self.assertEqual(videos[0].extractor, "f2")
        command = mock_run_command.call_args.args[0]
        self.assertEqual(command[:3], [fetch_videos.sys.executable, "-m", "f2"])
        self.assertIn("douyin", command)
        self.assertIn("https://www.douyin.com/user/abc", command)

    def test_list_douyin_videos_requires_backend_or_manifest(self) -> None:
        source = fetch_videos.SourceEntry(
            platform="dy",
            uid="abc",
            url="https://www.douyin.com/user/abc",
            name="douyin-author",
        )
        args = fetch_videos.parse_args([])

        with mock.patch("scripts.fetch_videos.run_command", side_effect=FileNotFoundError("f2")):
            with self.assertRaises(fetch_videos.FetchVideosError) as ctx:
                fetch_videos.list_douyin_videos(source, args)

        self.assertIn("f2", str(ctx.exception))

    @mock.patch("scripts.fetch_videos.run_command")
    def test_dry_run_process_source_reports_douyin_candidates(self, mock_run_command: mock.Mock) -> None:
        mock_run_command.return_value = mock.Mock(
            stdout=json.dumps(
                {
                    "entries": [
                        {"id": "x1", "title": "四川麻将夜战", "url": "https://www.bilibili.com/video/BV1"}
                    ]
                },
                ensure_ascii=False,
            )
        )
        source = fetch_videos.SourceEntry(
            platform="bili",
            uid="123",
            url="https://space.bilibili.com/123",
            name="demo",
        )
        args = fetch_videos.parse_args(["--dry-run", "--browser", ""])
        state = {"downloaded": {}}

        report = fetch_videos.process_source(source, args, state)

        self.assertEqual(report.downloaded, 0)
        self.assertEqual(report.dry_run_candidates, ["https://www.bilibili.com/video/BV1"])


if __name__ == "__main__":
    unittest.main()
