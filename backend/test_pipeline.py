import os
import cv2
import json
import asyncio
import numpy as np
import tempfile
from pathlib import Path
from backend.server import VideoChunkProcessor, SemanticVideoAuditor, format_timestamp, MatchItem, AuditResponseSchema

def create_synthetic_test_video(file_path: str, duration_sec: int = 180, fps: int = 15):
    """
    Generate a synthetic multi-minute CCTV test video (180 seconds = 3 minutes).
    Simulates dark CCTV scene with periodic motion pulses.
    """
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)
    print(f"Generating {duration_sec}s test video ({total_frames} frames)...")

    for i in range(total_frames):
        current_sec = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8) + 20 # dark ambient

        # Simulate timestamp HUD text
        ts_str = format_timestamp(current_sec)
        cv2.putText(frame, f"CAM-01 REC {ts_str}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Simulate movement events:
        # Event 1 at 20-30s
        # Event 2 at 75-85s
        # Event 3 at 140-155s
        if (20 <= current_sec <= 30) or (75 <= current_sec <= 85) or (140 <= current_sec <= 155):
            pos_x = int(((i % (fps * 10)) / (fps * 10)) * (width - 40)) + 20
            pos_y = 120
            # Draw moving object
            cv2.rectangle(frame, (pos_x, pos_y), (pos_x + 30, pos_y + 40), (0, 255, 128), -1)

        out.write(frame)

    out.release()
    print(f"Test video created successfully at: {file_path}")

async def run_pipeline_verification():
    temp_dir = Path(tempfile.gettempdir()) / "sentinel_test"
    temp_dir.mkdir(parents=True, exist_ok=True)
    test_video_path = str(temp_dir / "test_long_surveillance.mp4")

    # 1. Create 3-minute video
    create_synthetic_test_video(test_video_path, duration_sec=180, fps=10)

    # 2. Step A - Chunking & Downsampling
    processor = VideoChunkProcessor(chunk_duration_sec=60.0, sample_fps=1.0, motion_threshold=10.0)
    meta = processor.inspect_and_chunk(test_video_path, "test_long_surveillance.mp4")

    print("\n--- [Step A] Chunking & Mapping Table Verification ---")
    print(f"Duration: {meta['duration']:.2f}s, Total Chunks: {len(meta['chunks'])}")
    assert len(meta['chunks']) == 3, f"Expected 3 chunks for 180s video, got {len(meta['chunks'])}"

    for chunk in meta['chunks']:
        print(f"Chunk: {chunk.chunk_id} -> Start: {chunk.start_second}s, End: {chunk.end_second}s, Frames: {chunk.frame_count}")
        assert chunk.end_second > chunk.start_second

    # 3. Step B - Keyframe Sampling & Motion Detection
    print("\n--- [Step B] Motion Filtering & Frame Sampling Verification ---")
    all_matches = []
    auditor = SemanticVideoAuditor()

    for idx, chunk in enumerate(meta['chunks']):
        sampled = processor.sample_chunk_frames(test_video_path, chunk.start_second, chunk.end_second)
        motion_count = sum(1 for f in sampled if f.get("has_motion"))
        print(f"Chunk {chunk.chunk_id}: {len(sampled)} sampled frames, {motion_count} active motion frames")

        matches = await auditor.analyze_chunk(chunk, sampled, "Locate delivery person or vehicle")
        for m in matches:
            print(f"  Match: [{m.start_time} - {m.end_time}] ({m.start_seconds}s - {m.end_seconds}s) [{m.category}]: {m.description}")
            # Verify global timestamp mapping
            assert m.start_seconds >= chunk.start_second, f"Timestamp drift: {m.start_seconds} < {chunk.start_second}"
            assert m.end_seconds <= chunk.end_second + 5.0, f"Timestamp overshoot: {m.end_seconds} > {chunk.end_second}"
        all_matches.extend(matches)

    # 4. Strict Schema Validation
    print("\n--- [Step C] Strict JSON Schema Validation ---")
    response_payload = AuditResponseSchema(
        matches=all_matches,
        total_chunks=len(meta['chunks']),
        video_duration=meta['duration'],
        query="Locate delivery person or vehicle"
    )

    json_str = json.dumps(response_payload.model_dump(), indent=2)
    print(json_str[:500] + "\n...")

    assert len(response_payload.matches) > 0, "Expected matches to be found"
    for match in response_payload.matches:
        assert isinstance(match.start_seconds, (int, float))
        assert isinstance(match.end_seconds, (int, float))
        assert match.start_time and match.end_time
        assert match.category in ["PERSON", "VEHICLE", "ANOMALY", "SECURITY", "OBJECT"]

    print("\n>>> ALL PIPELINE AND ARCHITECTURE VERIFICATIONS PASSED SUCCESSFULLY! <<<\n")

if __name__ == "__main__":
    asyncio.run(run_pipeline_verification())
