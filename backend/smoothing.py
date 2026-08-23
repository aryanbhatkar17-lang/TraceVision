"""
Temporal Smoothing Algorithm for TraceVision Stress-Test Pipeline
=================================================================

Problem:
    When processing blurry, low-light, or night-time CCTV footage, the AI model's
    per-frame confidence score can momentarily dip below the accepted threshold
    (e.g., due to motion blur, compression artefacts, or infrared sensor glitches),
    even though the target subject is clearly visible in the frames immediately
    before and after.

    Without smoothing, the retrieval engine would return choppy, fragmented 1-second
    clips instead of smooth, continuous segments — crippling the investigator UX.

Solution:
    A **sliding-window temporal smoother** that operates purely on the numerical
    confidence scores output by the vector database (PostgreSQL / Milvus).

    1. For every frame whose score falls below `threshold`, we inspect a
       neighbourhood of `tolerance_window` frames on each side.
    2. If *both* the left and right neighbourhoods contain at least one frame
       that meets the threshold, we **override** the dip and mark this frame
       as valid.  This is the "bridge" operation.
    3. After the bridging pass, consecutive valid timestamps are **merged** into
       continuous `(start, end)` clip ranges.
    4. An optional `min_clip_duration` filter drops any clip shorter than a
       specified length (default 0 — keep everything) so the React dashboard
       never serves a 0.3-second fragment to an investigator.

Computational Complexity:
    O(N × W) where N = number of frames, W = tolerance_window.
    In practice W ≤ 5, so this is effectively O(N) — runs in microseconds
    even for hour-long footage at 1 fps (3 600 frames).

    ● Zero GPU / VRAM usage — pure floating-point arithmetic on CPU.
    ● No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------
@dataclass
class FrameScore:
    """A single frame's timestamp and its vector-similarity confidence score."""
    timestamp: float        # Absolute timestamp in seconds (from video start)
    score: float            # Similarity / confidence score in [0.0, 1.0]
    bridged: bool = False   # True if this frame was rescued by smoothing


@dataclass
class ClipRange:
    """A continuous temporal segment where the subject is deemed present."""
    start: float            # Clip start timestamp (seconds)
    end: float              # Clip end timestamp (seconds)
    frame_count: int = 0    # Number of frames that compose this clip
    avg_confidence: float = 0.0   # Mean confidence across the clip's frames
    bridged_count: int = 0  # How many frames in this clip were bridged

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 4)

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "frame_count": self.frame_count,
            "avg_confidence": round(self.avg_confidence, 4),
            "bridged_count": self.bridged_count,
        }


# ---------------------------------------------------------------------------
# Core Temporal Smoothing Algorithm
# ---------------------------------------------------------------------------
class TemporalSmoother:
    """
    Sliding-window temporal smoother for vector-similarity confidence scores.

    Parameters
    ----------
    threshold : float
        Minimum confidence score for a frame to be accepted outright.
        Frames below this are candidates for bridging. Default 0.75.
    tolerance_window : int
        Number of neighbouring frames to inspect on each side when deciding
        whether to bridge a low-confidence frame. Default 2.
    min_clip_duration : float
        Minimum clip duration in seconds. Clips shorter than this are
        discarded after merging. Set to 0 to keep all clips. Default 0.
    frame_gap_limit : float
        Maximum time gap (seconds) between two consecutive valid frames
        that can still be merged into the same clip. If the gap exceeds
        this, a new clip starts. Default 2.0.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        tolerance_window: int = 2,
        min_clip_duration: float = 0.0,
        frame_gap_limit: float = 2.0,
    ):
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if tolerance_window < 1:
            raise ValueError(f"tolerance_window must be >= 1, got {tolerance_window}")
        if min_clip_duration < 0:
            raise ValueError(f"min_clip_duration must be >= 0, got {min_clip_duration}")
        if frame_gap_limit <= 0:
            raise ValueError(f"frame_gap_limit must be > 0, got {frame_gap_limit}")

        self.threshold = threshold
        self.tolerance_window = tolerance_window
        self.min_clip_duration = min_clip_duration
        self.frame_gap_limit = frame_gap_limit

    # ------------------------------------------------------------------
    # Step 1: Parse raw input into FrameScore objects
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_input(
        frame_scores: List[Tuple[float, float]],
    ) -> List[FrameScore]:
        """Convert list of (timestamp, score) tuples to FrameScore objects."""
        return [FrameScore(timestamp=ts, score=sc) for ts, sc in frame_scores]

    # ------------------------------------------------------------------
    # Step 2: Bridging pass  (the core sliding-window logic)
    # ------------------------------------------------------------------
    def _bridge_pass(self, frames: List[FrameScore]) -> List[FrameScore]:
        """
        For each frame below threshold, check if both its left and right
        neighbourhoods (within `tolerance_window`) contain at least one
        frame that meets the threshold.  If so, mark it as bridged.

        This is the key insight:  a single blurry frame surrounded by
        confident detections is *not* an absence — it's sensor noise.
        """
        n = len(frames)
        for i in range(n):
            if frames[i].score >= self.threshold:
                # Already passes — no bridging needed
                continue

            # --- Left neighbourhood ---
            left_start = max(0, i - self.tolerance_window)
            left_ok = any(
                frames[j].score >= self.threshold
                for j in range(left_start, i)
            )

            # --- Right neighbourhood ---
            right_end = min(n, i + self.tolerance_window + 1)
            right_ok = any(
                frames[j].score >= self.threshold
                for j in range(i + 1, right_end)
            )

            if left_ok and right_ok:
                frames[i].bridged = True

        return frames

    # ------------------------------------------------------------------
    # Step 3: Collect valid timestamps
    # ------------------------------------------------------------------
    def _collect_valid(self, frames: List[FrameScore]) -> List[FrameScore]:
        """Return only frames that are either above threshold or bridged."""
        return [f for f in frames if f.score >= self.threshold or f.bridged]

    # ------------------------------------------------------------------
    # Step 4: Merge into continuous clip ranges
    # ------------------------------------------------------------------
    def _merge_clips(self, valid_frames: List[FrameScore]) -> List[ClipRange]:
        """
        Group consecutive valid frames into ClipRange objects.

        Two frames belong to the same clip if the temporal gap between them
        is ≤ `frame_gap_limit` seconds.
        """
        if not valid_frames:
            return []

        clips: List[ClipRange] = []
        current_group: List[FrameScore] = [valid_frames[0]]

        for prev, curr in zip(valid_frames, valid_frames[1:]):
            if (curr.timestamp - prev.timestamp) <= self.frame_gap_limit:
                current_group.append(curr)
            else:
                clips.append(self._group_to_clip(current_group))
                current_group = [curr]

        # Flush the last group
        clips.append(self._group_to_clip(current_group))

        return clips

    @staticmethod
    def _group_to_clip(group: List[FrameScore]) -> ClipRange:
        """Convert a contiguous list of FrameScore objects into a ClipRange."""
        scores = [f.score for f in group]
        return ClipRange(
            start=group[0].timestamp,
            end=group[-1].timestamp,
            frame_count=len(group),
            avg_confidence=sum(scores) / len(scores) if scores else 0.0,
            bridged_count=sum(1 for f in group if f.bridged),
        )

    # ------------------------------------------------------------------
    # Step 5: Filter by minimum duration
    # ------------------------------------------------------------------
    def _filter_short_clips(self, clips: List[ClipRange]) -> List[ClipRange]:
        """Drop clips shorter than `min_clip_duration`."""
        if self.min_clip_duration <= 0:
            return clips
        return [c for c in clips if c.duration >= self.min_clip_duration]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def smooth(
        self,
        frame_scores: List[Tuple[float, float]],
    ) -> List[ClipRange]:
        """
        Full temporal smoothing pipeline.

        Parameters
        ----------
        frame_scores : list of (timestamp_sec, confidence_score) tuples
            Raw per-frame confidence output from the vector database.

        Returns
        -------
        list of ClipRange
            Continuous clip segments where the subject is present.
        """
        if not frame_scores:
            return []

        frames = self._parse_input(frame_scores)
        frames = self._bridge_pass(frames)
        valid = self._collect_valid(frames)
        clips = self._merge_clips(valid)
        clips = self._filter_short_clips(clips)
        return clips

    def smooth_timestamps(
        self,
        frame_scores: List[Tuple[float, float]],
    ) -> List[float]:
        """
        Convenience wrapper — returns a flat list of valid timestamps
        (matching the signature of the original `apply_temporal_smoothing`).
        """
        if not frame_scores:
            return []

        frames = self._parse_input(frame_scores)
        frames = self._bridge_pass(frames)
        valid = self._collect_valid(frames)
        return [f.timestamp for f in valid]

    def get_diagnostics(
        self,
        frame_scores: List[Tuple[float, float]],
    ) -> dict:
        """
        Run the pipeline and return full diagnostic info (useful for the
        "Stress Testing & Edge Cases" section of the presentation).
        """
        if not frame_scores:
            return {"total_frames": 0, "clips": [], "summary": {}}

        frames = self._parse_input(frame_scores)
        frames = self._bridge_pass(frames)
        valid = self._collect_valid(frames)
        clips = self._merge_clips(valid)
        filtered = self._filter_short_clips(clips)

        above = sum(1 for f in frames if f.score >= self.threshold)
        bridged = sum(1 for f in frames if f.bridged)
        dropped = sum(1 for f in frames if f.score < self.threshold and not f.bridged)

        return {
            "total_frames": len(frames),
            "above_threshold": above,
            "bridged_frames": bridged,
            "dropped_frames": dropped,
            "clips_before_filter": len(clips),
            "clips_after_filter": len(filtered),
            "clips": [c.to_dict() for c in filtered],
            "config": {
                "threshold": self.threshold,
                "tolerance_window": self.tolerance_window,
                "min_clip_duration": self.min_clip_duration,
                "frame_gap_limit": self.frame_gap_limit,
            },
        }


# ---------------------------------------------------------------------------
# Legacy compatibility wrapper
# ---------------------------------------------------------------------------
def apply_temporal_smoothing(
    frame_scores: List[Tuple[float, float]],
    threshold: float = 0.75,
    tolerance_seconds: int = 2,
) -> List[float]:
    """
    Drop-in replacement for the original function signature.

    Returns a flat list of valid timestamps after smoothing.
    """
    smoother = TemporalSmoother(
        threshold=threshold,
        tolerance_window=tolerance_seconds,
    )
    return smoother.smooth_timestamps(frame_scores)


# ---------------------------------------------------------------------------
# Self-test suite
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print("=" * 70)
    print("  TraceVision -- Temporal Smoothing Algorithm  |  Self-Test Suite")
    print("=" * 70)

    smoother = TemporalSmoother(
        threshold=0.75,
        tolerance_window=2,
        min_clip_duration=0.0,
        frame_gap_limit=2.0,
    )

    # ------------------------------------------------------------------
    # Test 1: Single dip bridged
    # ------------------------------------------------------------------
    print("\n[Test 1] Single dip in the middle -- should be bridged")
    data_1 = [
        (73.0, 0.88),
        (74.0, 0.90),
        (75.0, 0.20),   # <- dip
        (76.0, 0.85),
        (77.0, 0.82),
    ]
    clips_1 = smoother.smooth(data_1)
    assert len(clips_1) == 1, f"Expected 1 clip, got {len(clips_1)}"
    assert clips_1[0].start == 73.0
    assert clips_1[0].end == 77.0
    assert clips_1[0].bridged_count == 1
    print(f"   [OK]  1 clip: {clips_1[0].start}s -> {clips_1[0].end}s  "
          f"(bridged {clips_1[0].bridged_count} frame)")

    # ------------------------------------------------------------------
    # Test 2: Two consecutive dips bridged
    # ------------------------------------------------------------------
    print("\n[Test 2] Two consecutive dips -- both should be bridged")
    data_2 = [
        (10.0, 0.80),
        (11.0, 0.30),   # <- dip
        (12.0, 0.25),   # <- dip
        (13.0, 0.82),
        (14.0, 0.90),
    ]
    clips_2 = smoother.smooth(data_2)
    assert len(clips_2) == 1, f"Expected 1 clip, got {len(clips_2)}"
    assert clips_2[0].bridged_count == 2
    print(f"   [OK]  1 clip: {clips_2[0].start}s -> {clips_2[0].end}s  "
          f"(bridged {clips_2[0].bridged_count} frames)")

    # ------------------------------------------------------------------
    # Test 3: Dip at the very beginning -- NOT bridged (no left neighbour)
    # ------------------------------------------------------------------
    print("\n[Test 3] Dip at the start -- no left neighbour, should NOT bridge")
    data_3 = [
        (0.0, 0.10),    # <- dip at start
        (1.0, 0.85),
        (2.0, 0.90),
    ]
    clips_3 = smoother.smooth(data_3)
    assert len(clips_3) == 1
    assert clips_3[0].start == 1.0, f"Expected clip to start at 1.0, got {clips_3[0].start}"
    print(f"   [OK]  1 clip: {clips_3[0].start}s -> {clips_3[0].end}s  "
          f"(frame at 0.0s correctly dropped)")

    # ------------------------------------------------------------------
    # Test 4: Dip at the very end -- NOT bridged (no right neighbour)
    # ------------------------------------------------------------------
    print("\n[Test 4] Dip at the end -- no right neighbour, should NOT bridge")
    data_4 = [
        (50.0, 0.80),
        (51.0, 0.92),
        (52.0, 0.15),   # <- dip at end
    ]
    clips_4 = smoother.smooth(data_4)
    assert len(clips_4) == 1
    assert clips_4[0].end == 51.0, f"Expected clip to end at 51.0, got {clips_4[0].end}"
    print(f"   [OK]  1 clip: {clips_4[0].start}s -> {clips_4[0].end}s  "
          f"(frame at 52.0s correctly dropped)")

    # ------------------------------------------------------------------
    # Test 5: Large gap -- two separate clips
    # ------------------------------------------------------------------
    print("\n[Test 5] Large gap between detections -- should produce 2 clips")
    data_5 = [
        (10.0, 0.90),
        (11.0, 0.88),
        # --- 8 second gap ---
        (19.0, 0.85),
        (20.0, 0.92),
    ]
    clips_5 = smoother.smooth(data_5)
    assert len(clips_5) == 2, f"Expected 2 clips, got {len(clips_5)}"
    print(f"   [OK]  2 clips: [{clips_5[0].start}-{clips_5[0].end}] "
          f"and [{clips_5[1].start}-{clips_5[1].end}]")

    # ------------------------------------------------------------------
    # Test 6: All frames below threshold -- no clips
    # ------------------------------------------------------------------
    print("\n[Test 6] All frames below threshold -- no clips")
    data_6 = [
        (0.0, 0.10),
        (1.0, 0.20),
        (2.0, 0.30),
        (3.0, 0.15),
    ]
    clips_6 = smoother.smooth(data_6)
    assert len(clips_6) == 0, f"Expected 0 clips, got {len(clips_6)}"
    print("   [OK]  0 clips (all noise, correctly discarded)")

    # ------------------------------------------------------------------
    # Test 7: Minimum clip duration filter
    # ------------------------------------------------------------------
    print("\n[Test 7] Minimum clip duration filter (3s) -- short clip dropped")
    strict_smoother = TemporalSmoother(
        threshold=0.75,
        tolerance_window=2,
        min_clip_duration=3.0,
        frame_gap_limit=2.0,
    )
    data_7 = [
        (10.0, 0.90),  # 1-second clip
        (11.0, 0.88),
        # --- gap ---
        (20.0, 0.92),  # 4-second clip
        (21.0, 0.85),
        (22.0, 0.80),
        (23.0, 0.88),
        (24.0, 0.91),
    ]
    clips_7 = strict_smoother.smooth(data_7)
    assert len(clips_7) == 1, f"Expected 1 clip (short one filtered), got {len(clips_7)}"
    assert clips_7[0].start == 20.0
    print(f"   [OK]  1 clip kept: {clips_7[0].start}s -> {clips_7[0].end}s  "
          f"(short clip at 10-11s correctly filtered)")

    # ------------------------------------------------------------------
    # Test 8: Night-time stress scenario -- many intermittent dips
    # ------------------------------------------------------------------
    print("\n[Test 8] Night-time stress scenario -- intermittent dips across 15 frames")
    data_night = [
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
        (113.0, 0.15),  # dip near end
        (114.0, 0.80),
    ]
    diag = smoother.get_diagnostics(data_night)
    clips_8 = diag["clips"]
    print(f"   Total frames: {diag['total_frames']}")
    print(f"   Above threshold: {diag['above_threshold']}")
    print(f"   Bridged frames:  {diag['bridged_frames']}")
    print(f"   Dropped frames:  {diag['dropped_frames']}")
    print(f"   Clips produced:  {diag['clips_after_filter']}")
    for c in diag["clips"]:
        print(f"      [{c['start']}s - {c['end']}s]  dur={c['duration']}s  "
              f"frames={c['frame_count']}  bridged={c['bridged_count']}  "
              f"avg_conf={c['avg_confidence']:.3f}")

    assert len(clips_8) == 1, f"Expected 1 continuous clip, got {len(clips_8)}"
    assert clips_8[0]["start"] == 100.0
    assert clips_8[0]["end"] == 114.0
    print("   [OK]  All dips bridged into 1 continuous clip!")

    # ------------------------------------------------------------------
    # Legacy compatibility test
    # ------------------------------------------------------------------
    print("\n[Test 9] Legacy API compatibility (apply_temporal_smoothing)")
    legacy_result = apply_temporal_smoothing(data_1, threshold=0.75, tolerance_seconds=2)
    assert legacy_result == [73.0, 74.0, 75.0, 76.0, 77.0], f"Unexpected: {legacy_result}"
    print(f"   [OK]  Kept timestamps: {legacy_result}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ALL 9 TESTS PASSED")
    print("=" * 70)
    print("\nDiagnostics JSON (Test 8 - Night Stress):")
    print(json.dumps(diag, indent=2))