"""
Comprehensive test suite for the Temporal Smoothing Algorithm.

Tests cover:
  - Bridging logic (single dips, consecutive dips, edge dips)
  - Clip merging (gap detection, continuity)
  - Minimum duration filter
  - Night-time / blurry stress scenarios
  - Parameter validation
  - Legacy API compatibility
  - Empty / degenerate inputs
  - Diagnostics output
"""

import pytest
from smoothing import TemporalSmoother, apply_temporal_smoothing, ClipRange


# =========================================================================
# Fixtures
# =========================================================================
@pytest.fixture
def default_smoother():
    """Standard smoother with default hackathon parameters."""
    return TemporalSmoother(
        threshold=0.75,
        tolerance_window=2,
        min_clip_duration=0.0,
        frame_gap_limit=2.0,
    )


@pytest.fixture
def strict_smoother():
    """Smoother that filters clips shorter than 3 seconds."""
    return TemporalSmoother(
        threshold=0.75,
        tolerance_window=2,
        min_clip_duration=3.0,
        frame_gap_limit=2.0,
    )


# =========================================================================
# Basic Bridging Tests
# =========================================================================
class TestBridging:
    def test_single_dip_is_bridged(self, default_smoother):
        """A single low frame surrounded by confident detections is bridged."""
        data = [
            (73.0, 0.88),
            (74.0, 0.90),
            (75.0, 0.20),   # dip
            (76.0, 0.85),
            (77.0, 0.82),
        ]
        clips = default_smoother.smooth(data)
        assert len(clips) == 1
        assert clips[0].start == 73.0
        assert clips[0].end == 77.0
        assert clips[0].bridged_count == 1

    def test_two_consecutive_dips_bridged(self, default_smoother):
        """Two consecutive dips within tolerance should both be bridged."""
        data = [
            (10.0, 0.80),
            (11.0, 0.30),   # dip
            (12.0, 0.25),   # dip
            (13.0, 0.82),
            (14.0, 0.90),
        ]
        clips = default_smoother.smooth(data)
        assert len(clips) == 1
        assert clips[0].bridged_count == 2

    def test_dip_at_start_not_bridged(self, default_smoother):
        """A dip at the very start has no left neighbour → NOT bridged."""
        data = [
            (0.0, 0.10),
            (1.0, 0.85),
            (2.0, 0.90),
        ]
        clips = default_smoother.smooth(data)
        assert len(clips) == 1
        assert clips[0].start == 1.0  # frame at 0.0 dropped

    def test_dip_at_end_not_bridged(self, default_smoother):
        """A dip at the very end has no right neighbour → NOT bridged."""
        data = [
            (50.0, 0.80),
            (51.0, 0.92),
            (52.0, 0.15),
        ]
        clips = default_smoother.smooth(data)
        assert len(clips) == 1
        assert clips[0].end == 51.0  # frame at 52.0 dropped

    def test_wide_dip_not_bridged(self):
        """A dip wider than the tolerance window should NOT be bridged."""
        smoother = TemporalSmoother(threshold=0.75, tolerance_window=1)
        data = [
            (10.0, 0.80),
            (11.0, 0.30),   # dip
            (12.0, 0.25),   # dip — too far from 10.0 with window=1
            (13.0, 0.82),
        ]
        clips = smoother.smooth(data)
        # frame 11 has left=10(ok), right=12(fail) → not bridged
        # frame 12 has left=11(fail), right=13(ok) → not bridged
        assert len(clips) == 2
        assert clips[0].end == 10.0
        assert clips[1].start == 13.0


# =========================================================================
# Clip Merging Tests
# =========================================================================
class TestClipMerging:
    def test_large_gap_produces_two_clips(self, default_smoother):
        """Frames separated by > frame_gap_limit → separate clips."""
        data = [
            (10.0, 0.90),
            (11.0, 0.88),
            # gap of 8 seconds
            (19.0, 0.85),
            (20.0, 0.92),
        ]
        clips = default_smoother.smooth(data)
        assert len(clips) == 2
        assert clips[0].end == 11.0
        assert clips[1].start == 19.0

    def test_frames_within_gap_limit_merged(self, default_smoother):
        """Frames within frame_gap_limit should be merged into one clip."""
        data = [
            (10.0, 0.90),
            (11.5, 0.88),  # gap=1.5 < 2.0
            (13.0, 0.85),  # gap=1.5 < 2.0
        ]
        clips = default_smoother.smooth(data)
        assert len(clips) == 1
        assert clips[0].start == 10.0
        assert clips[0].end == 13.0

    def test_three_distinct_clip_segments(self, default_smoother):
        """Three isolated groups of frames → three clips."""
        data = [
            (1.0, 0.80), (2.0, 0.82),
            # gap
            (10.0, 0.90), (11.0, 0.91),
            # gap
            (20.0, 0.85), (21.0, 0.86), (22.0, 0.88),
        ]
        clips = default_smoother.smooth(data)
        assert len(clips) == 3


# =========================================================================
# Duration Filter Tests
# =========================================================================
class TestDurationFilter:
    def test_short_clip_filtered(self, strict_smoother):
        """Clips shorter than min_clip_duration are dropped."""
        data = [
            (10.0, 0.90),
            (11.0, 0.88),
            # gap
            (20.0, 0.92),
            (21.0, 0.85),
            (22.0, 0.80),
            (23.0, 0.88),
            (24.0, 0.91),
        ]
        clips = strict_smoother.smooth(data)
        assert len(clips) == 1
        assert clips[0].start == 20.0  # only the longer clip survives

    def test_all_clips_filtered_if_too_short(self, strict_smoother):
        """If all clips are too short, result is empty."""
        data = [
            (10.0, 0.90),
            (11.0, 0.88),
            # gap
            (20.0, 0.85),
            (21.0, 0.80),
        ]
        clips = strict_smoother.smooth(data)
        assert len(clips) == 0


# =========================================================================
# Edge Cases & Degenerate Inputs
# =========================================================================
class TestEdgeCases:
    def test_empty_input(self, default_smoother):
        """Empty input returns empty clips."""
        assert default_smoother.smooth([]) == []
        assert default_smoother.smooth_timestamps([]) == []

    def test_single_frame_above_threshold(self, default_smoother):
        """Single frame above threshold → 1 clip of duration 0."""
        clips = default_smoother.smooth([(5.0, 0.90)])
        assert len(clips) == 1
        assert clips[0].duration == 0.0

    def test_single_frame_below_threshold(self, default_smoother):
        """Single frame below threshold → no clips."""
        clips = default_smoother.smooth([(5.0, 0.40)])
        assert len(clips) == 0

    def test_all_frames_above_threshold(self, default_smoother):
        """All frames above threshold → 1 clip, 0 bridged."""
        data = [(i, 0.90) for i in range(10)]
        clips = default_smoother.smooth(data)
        assert len(clips) == 1
        assert clips[0].bridged_count == 0
        assert clips[0].frame_count == 10

    def test_all_frames_below_threshold(self, default_smoother):
        """All frames below threshold → no clips."""
        data = [(i, 0.10) for i in range(10)]
        clips = default_smoother.smooth(data)
        assert len(clips) == 0

    def test_exact_threshold_is_included(self, default_smoother):
        """Frames exactly AT threshold should pass (>=, not >)."""
        data = [(1.0, 0.75), (2.0, 0.75)]
        clips = default_smoother.smooth(data)
        assert len(clips) == 1
        assert clips[0].frame_count == 2


# =========================================================================
# Night-Time Stress Scenario
# =========================================================================
class TestNightTimeStress:
    def test_intermittent_dips_across_15_frames(self, default_smoother):
        """
        Simulates a night-time feed where the IR sensor causes
        intermittent confidence drops across 15 frames.
        The algorithm should bridge all dips into 1 continuous clip.
        """
        data = [
            (100.0, 0.82),
            (101.0, 0.78),
            (102.0, 0.55),  # dip
            (103.0, 0.80),
            (104.0, 0.40),  # dip
            (105.0, 0.76),
            (106.0, 0.30),  # dip
            (107.0, 0.81),
            (108.0, 0.85),
            (109.0, 0.20),  # dip
            (110.0, 0.83),
            (111.0, 0.88),
            (112.0, 0.90),
            (113.0, 0.15),  # dip
            (114.0, 0.80),
        ]
        clips = default_smoother.smooth(data)
        assert len(clips) == 1
        assert clips[0].start == 100.0
        assert clips[0].end == 114.0
        assert clips[0].bridged_count == 5  # all 5 dips bridged

    def test_motion_blur_burst(self, default_smoother):
        """
        Simulates a camera auto-focus failure — 3 consecutive bad frames
        within a window=2 tolerance. Middle frame can't see far enough.
        """
        data = [
            (30.0, 0.85),
            (31.0, 0.40),  # dip - left=30(ok), right=32(fail)/33(ok) → bridged
            (32.0, 0.35),  # dip - left=30(ok)/31(fail), right=33(ok)/34(ok) → bridged
            (33.0, 0.30),  # dip - left=31(fail)/32(fail), right=34(ok) → NOT bridged
            (34.0, 0.88),
            (35.0, 0.90),
        ]
        clips = default_smoother.smooth(data)
        # Frame 31: left has 30(ok) → yes. Right has 32(fail), 33(fail) → no. NOT bridged.
        # Frame 32: left has 30(ok), 31(fail) → yes. Right has 33(fail), 34(ok) → yes. BRIDGED.
        # Frame 33: left has 31(fail), 32(fail) → no (32 score is 0.35<0.75). NOT bridged.
        # Actually: frame 32 is bridged in the second pass, so left for 33 = [31(0.40), 32(0.35)]
        # — neither >= 0.75. So 33 not bridged.
        # Frame 31: right neighbourhood = [32(0.35), 33(0.30)] — neither >= 0.75 → NOT bridged
        # So result: clips from [30] and [32, 34, 35] — 32 is bridged
        # Wait, frame 31: left=[30(0.85)] ✓, right=[32(0.35), 33(0.30)] ✗ → NOT bridged
        # frame 32: left=[30(0.85), 31(0.40)] → 30 ✓, right=[33(0.30), 34(0.88)] → 34 ✓ → BRIDGED
        # frame 33: left=[31(0.40), 32(0.35)] → neither ✓ → NOT bridged
        # Valid: 30, 32(bridged), 34, 35
        # Gaps: 30→32 = 2.0 ≤ 2.0 → same clip. 32→34 = 2.0 ≤ 2.0 → same clip.
        assert len(clips) == 1
        assert clips[0].bridged_count == 1  # only frame 32


# =========================================================================
# Parameter Validation Tests
# =========================================================================
class TestParameterValidation:
    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold"):
            TemporalSmoother(threshold=1.5)

    def test_invalid_tolerance_raises(self):
        with pytest.raises(ValueError, match="tolerance_window"):
            TemporalSmoother(tolerance_window=0)

    def test_negative_min_clip_raises(self):
        with pytest.raises(ValueError, match="min_clip_duration"):
            TemporalSmoother(min_clip_duration=-1.0)

    def test_zero_gap_limit_raises(self):
        with pytest.raises(ValueError, match="frame_gap_limit"):
            TemporalSmoother(frame_gap_limit=0)


# =========================================================================
# Diagnostics Output
# =========================================================================
class TestDiagnostics:
    def test_diagnostics_structure(self, default_smoother):
        data = [
            (10.0, 0.90),
            (11.0, 0.30),
            (12.0, 0.85),
        ]
        diag = default_smoother.get_diagnostics(data)

        assert diag["total_frames"] == 3
        assert diag["above_threshold"] == 2
        assert diag["bridged_frames"] == 1
        assert diag["dropped_frames"] == 0
        assert "clips" in diag
        assert "config" in diag
        assert diag["config"]["threshold"] == 0.75

    def test_empty_diagnostics(self, default_smoother):
        diag = default_smoother.get_diagnostics([])
        assert diag["total_frames"] == 0
        assert diag["clips"] == []


# =========================================================================
# Legacy Compatibility
# =========================================================================
class TestLegacyAPI:
    def test_apply_temporal_smoothing_returns_timestamps(self):
        data = [
            (73.0, 0.88),
            (74.0, 0.90),
            (75.0, 0.20),  # dip
            (76.0, 0.85),
            (77.0, 0.82),
        ]
        result = apply_temporal_smoothing(data, threshold=0.75, tolerance_seconds=2)
        assert result == [73.0, 74.0, 75.0, 76.0, 77.0]

    def test_legacy_empty_input(self):
        assert apply_temporal_smoothing([]) == []


# =========================================================================
# ClipRange Data Model
# =========================================================================
class TestClipRange:
    def test_duration_property(self):
        c = ClipRange(start=10.0, end=15.0, frame_count=5)
        assert c.duration == 5.0

    def test_to_dict(self):
        c = ClipRange(start=1.0, end=4.0, frame_count=3, avg_confidence=0.85, bridged_count=1)
        d = c.to_dict()
        assert d["start"] == 1.0
        assert d["end"] == 4.0
        assert d["duration"] == 3.0
        assert d["frame_count"] == 3
        assert d["bridged_count"] == 1
