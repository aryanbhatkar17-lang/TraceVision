import os
import cv2
import json
import time
import asyncio
import numpy as np
import tempfile
from pathlib import Path
from backend.server import VideoChunkProcessor, SemanticVideoAuditor, format_timestamp, UPLOAD_DIR

def create_2min_cctv_video(output_path: str, duration_sec: int = 150, fps: int = 10):
    """
    Generate a synthetic 2.5-minute CCTV video (150 seconds = 1500 frames).
    Contains multiple simulated movement events across chunks:
    - Event 1: 00:15 - 00:25 (Chunk 1)
    - Event 2: 01:10 - 01:25 (Chunk 2)
    - Event 3: 02:05 - 02:20 (Chunk 3)
    """
    width, height = 480, 270
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)
    print(f"Generating {duration_sec}s CCTV video ({total_frames} frames at {fps} fps)...")

    for i in range(total_frames):
        current_sec = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8) + 25

        # Render simulated CCTV overlay
        ts_str = format_timestamp(current_sec)
        cv2.putText(frame, f"REC // CAM-02 N_TERMINAL {ts_str}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)

        # Draw moving objects during activity intervals
        if (15 <= current_sec <= 25) or (70 <= current_sec <= 85) or (125 <= current_sec <= 140):
            pos_x = int(((i % (fps * 8)) / (fps * 8)) * (width - 50)) + 20
            pos_y = 130
            cv2.rectangle(frame, (pos_x, pos_y), (pos_x + 35, pos_y + 45), (200, 200, 50), -1)

        out.write(frame)

    out.release()
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Video created: {output_path} ({file_size_mb:.2f} MB)")
    return file_size_mb

async def test_large_video_pipeline():
    print("\n=======================================================")
    print(">>> TESTING 2+ MINUTE VIDEO INGESTION & OPTIMIZATION <<<")
    print("=======================================================")

    temp_test_dir = Path(tempfile.gettempdir()) / "sentinel_2min_test"
    temp_test_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(temp_test_dir / "feed_2min30s.mp4")

    # 1. Generate 2.5 min video
    create_2min_cctv_video(video_path, duration_sec=150, fps=10)

    # 2. Inspect & Chunk (Chronological 60s Chunks)
    processor = VideoChunkProcessor(
        chunk_duration_sec=60.0,
        sample_fps=1.0,
        target_width=640,
        target_height=360,
        jpeg_quality=75
    )
    meta = processor.inspect_and_chunk(video_path, "feed_2min30s.mp4")

    print(f"\n[Step 1] Metadata & Chunking:")
    print(f"  Duration: {meta['duration']:.2f}s | FPS: {meta['fps']} | Total Chunks: {len(meta['chunks'])}")
    assert len(meta['chunks']) == 3, f"Expected 3 chunks for 150s video, got {len(meta['chunks'])}"

    # 3. Keyframe Extraction & JPEG Compression Test
    print(f"\n[Step 2] 1 FPS Keyframe Extraction & JPEG Compression:")
    total_extracted_frames = 0
    total_compressed_bytes = 0

    auditor = SemanticVideoAuditor()
    all_matches = []

    for chunk in meta['chunks']:
        keyframes = processor.sample_and_compress_chunk_keyframes(video_path, chunk.start_second, chunk.end_second)
        total_extracted_frames += len(keyframes)
        chunk_bytes = sum(k['jpeg_size_bytes'] for k in keyframes)
        total_compressed_bytes += chunk_bytes

        print(f"  Chunk {chunk.chunk_id} [{chunk.start_second}s - {chunk.end_second}s]: "
              f"{len(keyframes)} keyframes, compressed size: {chunk_bytes / 1024:.1f} KB")

        # Verify exact frame_idx / fps = timestamp_seconds mapping
        for k in keyframes:
            expected_sec = round(k['frame_idx'] / meta['fps'], 2)
            assert k['timestamp_sec'] == expected_sec, f"Drift: {k['timestamp_sec']} != {expected_sec}"

        matches = await auditor.analyze_chunk(chunk, keyframes, "Locate delivery person or vehicle")
        all_matches.extend(matches)

    print(f"  Total Extracted Keyframes: {total_extracted_frames} (avoids sending 1500 raw frames)")
    print(f"  Total Compressed Vision Payload: {total_compressed_bytes / (1024 * 1024):.2f} MB")

    # 4. Strict Schema & Global Timestamp Validation
    print(f"\n[Step 3] Strict Schema & Global Timestamp Verification:")
    assert len(all_matches) > 0, "Expected matches to be detected across chunks"

    for idx, m in enumerate(all_matches):
        print(f"  Match #{idx+1}: [{m.start_time} - {m.end_time}] ({m.start_seconds}s - {m.end_seconds}s) "
              f"[{m.category}]: {m.description}")
        assert m.start_seconds >= 0 and m.end_seconds <= meta['duration'] + 5.0
        assert m.category in ["PERSON", "VEHICLE", "ANOMALY", "SECURITY", "OBJECT"]

    print("\n>>> 2-MINUTE VIDEO INGESTION & PROCESSING TEST PASSED SUCCESSFULLY! <<<\n")

if __name__ == "__main__":
    asyncio.run(test_large_video_pipeline())
