from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from mahjong_rt.game_events import GameEvent
from mahjong_rt.offline_game_events import OfflineEventConfig, reconstruct_events


def _tile(track_id: int, label: str, x: float, y: float = 100.0) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "label": label,
        "confidence": 0.9,
        "bbox": [x, y, 40.0, 50.0],
        "zone": "river",
        "state": "CONFIRMED",
        "visible": True,
    }


def _summaries(changes: list[tuple[float, list[dict[str, Any]]]], duration: float = 5.0) -> list[dict[str, Any]]:
    result = []
    active: list[dict[str, Any]] = []
    change_idx = 0
    for frame in range(int(duration * 10) + 1):
        ts = frame / 10
        while change_idx < len(changes) and changes[change_idx][0] <= ts:
            active = changes[change_idx][1]
            change_idx += 1
        result.append({"type": "frame_summary", "frame_idx": frame, "ts": ts, "tiles": active})
    return result


def _config(**kwargs: Any) -> OfflineEventConfig:
    defaults = {
        "initial_window_s": 0.5,
        "stable_window_s": 0.5,
        "min_presence_ratio": 0.6,
        "sample_step_s": 0.1,
        "min_gap_s": 0.6,
        "start_player": "me",
    }
    defaults.update(kwargs)
    return OfflineEventConfig(**defaults)


def test_initial_tiles_do_not_create_events() -> None:
    summaries = _summaries([(0.0, [_tile(1, "b1", 100)])])
    assert reconstruct_events(summaries, config=_config()) == []


def test_persistent_new_tile_creates_one_discard_within_first_three_seconds() -> None:
    initial = [_tile(1, "b1", 100)]
    after = initial + [_tile(2, "t5", 220)]
    events = reconstruct_events(_summaries([(0.0, initial), (1.0, after)]), config=_config())
    assert [(event.event_type, event.player, event.tile) for event in events] == [("discard", "me", "t5")]
    assert isinstance(events[0], GameEvent)
    assert 0.8 <= events[0].ts <= 1.5
    assert events[0].to_dict()["seq"] == 1


def test_initial_tile_confirmed_late_in_initial_window_is_not_an_event() -> None:
    delayed_initial = [_tile(1, "b1", 100)]
    summaries = _summaries([(0.0, []), (0.4, delayed_initial)])
    assert reconstruct_events(summaries, config=_config()) == []


def test_single_frame_flicker_is_ignored() -> None:
    initial = [_tile(1, "b1", 100)]
    flicker = initial + [_tile(2, "t5", 220)]
    summaries = _summaries([(0.0, initial), (1.0, flicker), (1.1, initial)])
    assert reconstruct_events(summaries, config=_config()) == []


def test_track_id_change_at_same_position_is_not_new_tile() -> None:
    before = [_tile(1, "b1", 100)]
    after = [_tile(99, "b1", 102)]
    summaries = _summaries([(0.0, before), (1.0, after)])
    assert reconstruct_events(summaries, config=_config()) == []


def test_two_new_tiles_are_deduplicated_and_follow_turn_order() -> None:
    first = [_tile(1, "b1", 100)]
    second = first + [_tile(2, "t5", 220)]
    third = second + [_tile(3, "w7", 340)]
    summaries = _summaries([(0.0, first), (1.0, second), (3.0, third)], duration=5.0)
    events = reconstruct_events(summaries, config=_config())
    assert [(event.player, event.tile) for event in events] == [("me", "t5"), ("right", "w7")]


def test_fifth_copy_of_label_is_rejected() -> None:
    initial = [_tile(idx, "b1", idx * 100) for idx in range(1, 5)]
    fifth = initial + [_tile(5, "b1", 500)]
    events = reconstruct_events(
        _summaries([(0.0, initial), (1.0, fifth)]),
        config=_config(stable_window_s=0.3),
    )
    assert events == []


def test_filters_non_river_invisible_and_unlabelled_tiles() -> None:
    invalid = [
        {**_tile(1, "b1", 100), "zone": "seat_left"},
        {**_tile(2, "b2", 200), "visible": False},
        {**_tile(3, "b3", 300), "label": ""},
    ]
    summaries = _summaries([(0.0, []), (1.0, invalid)])
    assert reconstruct_events(summaries, config=_config()) == []


def test_track_id_switch_inside_window_still_forms_one_stable_tile() -> None:
    summaries = _summaries([(0.0, []), (0.8, [_tile(1, "t5", 220)]), (1.1, [_tile(2, "t5", 221)])])
    events = reconstruct_events(summaries, config=_config(stable_window_s=0.6))
    assert [event.tile for event in events] == ["t5"]


def test_camera_motion_is_cancelled_by_homography() -> None:
    summaries = _summaries([(0.0, [_tile(1, "b1", 100)])])
    homographies = []
    for index, summary in enumerate(summaries):
        shift = 0.0 if index == 0 else 2.0
        summary["tiles"] = [_tile(1, "b1", 100 + index * 2)]
        homographies.append([[1.0, 0.0, shift], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert reconstruct_events(summaries, homographies, _config()) == []


def test_rejects_unsorted_summaries_when_homographies_are_incremental() -> None:
    summaries = _summaries([(0.0, [_tile(1, "b1", 100)])], duration=1.0)
    summaries[0], summaries[1] = summaries[1], summaries[0]
    homographies = [None] * len(summaries)
    try:
        reconstruct_events(summaries, homographies, _config())
    except ValueError as error:
        assert "sorted" in str(error)
    else:
        raise AssertionError("unsorted incremental homographies must be rejected")


def test_eval_cli_accepts_stable_and_online_methods(tmp_path: Path, monkeypatch: Any) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    sys.modules.pop("eval_game_events", None)
    from eval_game_events import align, main, similarity

    pred = {"event_type": "discard", "ts": 1.0, "tile": "wrong", "player": "wrong"}
    truth = {"type": "discard", "t": 1.0, "tile": "b1", "who": "me"}
    assert similarity(pred, truth) == 1.0
    assert align([pred], [truth]) == [(0, 0)]

    testset = tmp_path / "testset"
    testset.mkdir()
    (testset / "events_gt.json").write_text('{"clips": {}}', encoding="utf-8")
    config = tmp_path / "pipeline.yaml"
    config.write_text("{}", encoding="utf-8")

    assert main(["--testset", str(testset), "--config", str(config), "--method", "backtrack"]) == 0
    assert main(["--testset", str(testset), "--config", str(config), "--method", "stable"]) == 0
    assert main(["--testset", str(testset), "--config", str(config), "--method", "online"]) == 0
