"""
Integration Tests for TraceVision API — Search, Smooth & Clip Pipeline
======================================================================
Tests cover:
  - POST /api/search       → vector search + temporal smoothing
  - POST /api/clip         → FFmpeg / OpenCV clip cutting
  - GET  /api/hardware     → resource snapshot
  - GET  /api/health       → health check
  - End-to-end: upload → search → clip pipeline
"""

import os
import sys
import json
import asyncio
import tempfile
import numpy as np
import cv2
from pathlib import Path

import pytest
import httpx

# Ensure backend/ is importable
_backend_dir = Path(__file__).resolve().parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from api import api, UPLOAD_DIR, monitor, cutter, smoother


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def test_video_path():
    """Create a synthetic 30-second CCTV test video for pipeline testing."""
    video_dir = UPLOAD_DIR
    video_dir.mkdir(parents=True, exist_ok=True)
    video_name = "test_api_video.mp4"
    video_path = video_dir / video_name

    if video_path.exists():
        return video_name

    width, height, fps, duration = 320, 240, 10, 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(int(duration * fps)):
        current_sec = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8) + 20

        # Add motion at 5-15s and 20-25s
        if 5 <= current_sec <= 15 or 20 <= current_sec <= 25:
            x = int((i % (fps * 5)) / (fps * 5) * (width - 40)) + 20
            cv2.rectangle(frame, (x, 100), (x + 30, 140), (0, 255, 128), -1)

        # Timestamp overlay
        cv2.putText(frame, f"T={current_sec:.1f}s", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        out.write(frame)

    out.release()
    return video_name


@pytest.fixture(scope="module")
def client():
    """Create an async test client for the API."""
    transport = httpx.ASGITransport(app=api)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# Health & Hardware Tests
# ---------------------------------------------------------------------------
class TestHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """GET /api/health should return healthy status."""
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "tracevision-api"
        assert "hardware" in data
        assert "ffmpeg_available" in data

    @pytest.mark.asyncio
    async def test_hardware_endpoint(self, client):
        """GET /api/hardware should return resource snapshot."""
        resp = await client.get("/api/hardware")
        assert resp.status_code == 200
        data = resp.json()
        assert "snapshot" in data
        assert "vram_ceiling_mb" in data
        assert data["vram_ceiling_mb"] == 6144.0
        assert "safe_to_proceed" in data
        snapshot = data["snapshot"]
        assert "cpu_percent" in snapshot
        assert "ram_used_mb" in snapshot
        assert "ram_total_mb" in snapshot
        assert "is_gpu_available" in snapshot

    @pytest.mark.asyncio
    async def test_hardware_headers_in_response(self, client):
        """Every response should include hardware resource headers."""
        resp = await client.get("/api/health")
        assert "x-ram-used-mb" in resp.headers
        assert "x-cpu-percent" in resp.headers


# ---------------------------------------------------------------------------
# Search Pipeline Tests
# ---------------------------------------------------------------------------
class TestSearch:
    @pytest.mark.asyncio
    async def test_search_basic(self, client, test_video_path):
        """POST /api/search should return smoothed clips."""
        resp = await client.post(
            "/api/search",
            json={"query": "person walking", "video_id": test_video_path, "top_k": 5}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "person walking"
        assert data["video_id"] == test_video_path
        assert data["video_duration"] > 0
        assert data["total_chunks"] >= 1
        assert "smoothed_clips" in data
        assert "diagnostics" in data
        assert "matches" in data  # backward-compatible
        assert data["diagnostics"]["total_frames"] > 0

    @pytest.mark.asyncio
    async def test_search_smoothed_clips_structure(self, client, test_video_path):
        """Smoothed clips should have correct fields."""
        resp = await client.post(
            "/api/search",
            json={"query": "vehicle detected", "video_id": test_video_path}
        )
        data = resp.json()
        for clip in data.get("smoothed_clips", []):
            assert "clip_id" in clip
            assert "start" in clip
            assert "end" in clip
            assert "duration" in clip
            assert clip["end"] > clip["start"]
            assert clip["duration"] > 0
            assert "avg_confidence" in clip
            assert "start_time" in clip
            assert "end_time" in clip

    @pytest.mark.asyncio
    async def test_search_nonexistent_video(self, client):
        """Searching a missing video should return 404."""
        resp = await client.post(
            "/api/search",
            json={"query": "test", "video_id": "does_not_exist.mp4"}
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_search_form_endpoint(self, client, test_video_path):
        """POST /api/search/form (form-encoded) should work identically."""
        resp = await client.post(
            "/api/search/form",
            data={"query": "suspicious activity", "video_id": test_video_path, "top_k": "3"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "smoothed_clips" in data


# ---------------------------------------------------------------------------
# Clip Cutting Tests
# ---------------------------------------------------------------------------
class TestClipCutter:
    @pytest.mark.asyncio
    async def test_cut_clip_basic(self, client, test_video_path):
        """POST /api/clip should cut a clip from the video."""
        resp = await client.post(
            "/api/clip",
            json={"video_id": test_video_path, "start": 2.0, "end": 8.0}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["start_sec"] == 2.0
        assert data["end_sec"] == 8.0
        assert data["duration_sec"] == 6.0
        assert data["method"] in ("ffmpeg_copy", "ffmpeg_reencode", "opencv")
        assert data["file_size_bytes"] > 0
        assert "download_url" in data

    @pytest.mark.asyncio
    async def test_cut_clip_invalid_range(self, client, test_video_path):
        """Clip with start >= end should return 400."""
        resp = await client.post(
            "/api/clip",
            json={"video_id": test_video_path, "start": 10.0, "end": 5.0}
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_cut_clip_nonexistent_video(self, client):
        """Clip from missing video should return 404."""
        resp = await client.post(
            "/api/clip",
            json={"video_id": "missing.mp4", "start": 0.0, "end": 5.0}
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_download_cut_clip(self, client, test_video_path):
        """Download a previously cut clip via GET /api/clip/download/{filename}."""
        # First cut a clip
        resp = await client.post(
            "/api/clip",
            json={"video_id": test_video_path, "start": 1.0, "end": 4.0}
        )
        assert resp.status_code == 200
        download_url = resp.json()["download_url"]

        # Then download it
        dl_resp = await client.get(download_url)
        assert dl_resp.status_code == 200
        assert dl_resp.headers.get("content-type") == "video/mp4"


# ---------------------------------------------------------------------------
# Hardware Monitor Unit Tests
# ---------------------------------------------------------------------------
class TestHardwareMonitor:
    def test_resource_snapshot(self):
        """HardwareMonitor.check_resources() should return valid snapshot."""
        snapshot = monitor.check_resources()
        assert snapshot.ram_used_mb > 0
        assert snapshot.ram_total_mb > 0
        assert snapshot.cpu_percent >= 0

    def test_safe_to_proceed(self):
        """is_safe_to_proceed should return True on a normal system."""
        assert monitor.is_safe_to_proceed(required_mb=0) is True

    def test_status_headers(self):
        """get_status_headers should include RAM and CPU."""
        headers = monitor.get_status_headers()
        assert "X-RAM-Used-MB" in headers
        assert "X-CPU-Percent" in headers


# ---------------------------------------------------------------------------
# Temporal Smoother Integration Test
# ---------------------------------------------------------------------------
class TestSmootherIntegration:
    def test_smoother_produces_clips(self):
        """TemporalSmoother should produce continuous clips from noisy data."""
        # Simulated night-CCTV frame scores
        frame_scores = [
            (100.0, 0.82), (101.0, 0.78), (102.0, 0.55),
            (103.0, 0.80), (104.0, 0.40), (105.0, 0.76),
            (106.0, 0.30), (107.0, 0.81), (108.0, 0.85),
        ]
        clips = smoother.smooth(frame_scores)
        assert len(clips) >= 1
        # Should bridge the dips into a continuous clip
        assert clips[0].start == 100.0

    def test_smoother_empty_input(self):
        """Empty input should return empty list."""
        clips = smoother.smooth([])
        assert clips == []


# ---------------------------------------------------------------------------
# ClipCutter Unit Tests
# ---------------------------------------------------------------------------
class TestClipCutterUnit:
    def test_cutter_init(self):
        """ClipCutter should initialize and report FFmpeg status."""
        assert isinstance(cutter.ffmpeg_available, bool)
        assert cutter.output_dir.exists()

    def test_cutter_invalid_source(self):
        """Cutting a nonexistent file should return failure."""
        result = cutter.cut_clip("nonexistent.mp4", 0.0, 5.0)
        assert result.success is False
        assert "does not exist" in result.error

    def test_cutter_invalid_range(self):
        """start >= end should return failure."""
        result = cutter.cut_clip("dummy.mp4", 10.0, 5.0)
        assert result.success is False


# ---------------------------------------------------------------------------
# End-to-End Pipeline Test
# ---------------------------------------------------------------------------
class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, client, test_video_path):
        """
        Full pipeline: search → get smoothed clips → cut first clip.
        """
        # Step 1: Search
        search_resp = await client.post(
            "/api/search",
            json={"query": "person with backpack", "video_id": test_video_path}
        )
        assert search_resp.status_code == 200
        search_data = search_resp.json()

        # Step 2: Verify diagnostics
        diag = search_data["diagnostics"]
        assert diag["total_frames"] > 0
        assert "smoother_config" in diag
        assert diag["smoother_config"]["threshold"] == 0.75

        # Step 3: If we have clips or matches, try to cut one
        clips = search_data.get("smoothed_clips", [])
        matches = search_data.get("matches", [])

        if clips:
            target = clips[0]
            cut_resp = await client.post(
                "/api/clip",
                json={
                    "video_id": test_video_path,
                    "start": target["start"],
                    "end": target["end"],
                }
            )
            assert cut_resp.status_code == 200
            assert cut_resp.json()["file_size_bytes"] > 0
        elif matches:
            target = matches[0]
            cut_resp = await client.post(
                "/api/clip",
                json={
                    "video_id": test_video_path,
                    "start": target["start_seconds"],
                    "end": target["end_seconds"],
                }
            )
            assert cut_resp.status_code == 200


# ---------------------------------------------------------------------------
# Run with: python -m pytest backend/test_api.py -v
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
