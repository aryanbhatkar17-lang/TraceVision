"""
TraceVision CCTV Surveillance System - Video Clip Cutter
SIH-2026 Hackathon Project

This module provides a robust utility for extracting sub-clips from MP4 videos.
It uses an FFmpeg-first approach with an OpenCV fallback. 

The primary extraction method uses FFmpeg's stream copy (`-c copy`), which is incredibly
fast and uses zero VRAM/CPU for transcoding, as it just repackages the existing frames.
If the fast copy fails (e.g., due to missing keyframes near the cut points resulting in
empty or unplayable files), it automatically falls back to re-encoding the segment
(`-c:v libx264 -preset ultrafast -crf 23 -c:a aac`).

If FFmpeg is not installed or not available on the system PATH, the module falls back
to a pure OpenCV implementation that reads and writes frames individually.
"""

import os
import subprocess
import logging
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Union, Dict, Any

import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClipResult:
    source_path: str
    output_path: str
    start_sec: float
    end_sec: float
    duration_sec: float
    method: str
    success: bool
    error: Optional[str]
    file_size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ClipCutter:
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        if output_dir is None:
            self.output_dir = Path(tempfile.gettempdir()) / "video_audit" / "clips"
        else:
            self.output_dir = Path(output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ffmpeg_path = self._find_ffmpeg()

    def _find_ffmpeg(self) -> Optional[str]:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            try:
                subprocess.run([ffmpeg_path, "-version"], check=True, capture_output=True)
                return ffmpeg_path
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        return None

    @property
    def ffmpeg_available(self) -> bool:
        return self._ffmpeg_path is not None

    def cut_clip(self, source_path: str, start_sec: float, end_sec: float, output_filename: Optional[str] = None) -> ClipResult:
        source = Path(source_path)
        if not source.exists():
            return self._error_result(str(source), "", start_sec, end_sec, "Source file does not exist")
            
        if start_sec >= end_sec:
            return self._error_result(str(source), "", start_sec, end_sec, "start_sec must be less than end_sec")

        if output_filename is None:
            output_filename = f"clip_{start_sec:.1f}s_{end_sec:.1f}s.mp4"
            
        output_path = self.output_dir / output_filename
        duration = end_sec - start_sec

        if self.ffmpeg_available:
            # Try FFmpeg copy
            result = self._try_ffmpeg_copy(str(source), str(output_path), start_sec, end_sec)
            if result.success:
                return result
                
            # Fallback to FFmpeg reencode
            logger.warning(f"FFmpeg copy failed for {source_path}, trying re-encode: {result.error}")
            result = self._try_ffmpeg_reencode(str(source), str(output_path), start_sec, end_sec)
            if result.success:
                return result
                
            logger.warning(f"FFmpeg re-encode failed for {source_path}, falling back to OpenCV: {result.error}")

        # Fallback to OpenCV
        return self._try_opencv(str(source), str(output_path), start_sec, end_sec)

    def _try_ffmpeg_copy(self, source: str, output: str, start_sec: float, end_sec: float) -> ClipResult:
        cmd = [
            self._ffmpeg_path, "-y",
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-i", source,
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output
        ]
        return self._run_ffmpeg_command(cmd, source, output, start_sec, end_sec, "ffmpeg_copy")

    def _try_ffmpeg_reencode(self, source: str, output: str, start_sec: float, end_sec: float) -> ClipResult:
        cmd = [
            self._ffmpeg_path, "-y",
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-i", source,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-c:a", "aac",
            output
        ]
        return self._run_ffmpeg_command(cmd, source, output, start_sec, end_sec, "ffmpeg_reencode")

    def _run_ffmpeg_command(self, cmd: List[str], source: str, output: str, start_sec: float, end_sec: float, method: str) -> ClipResult:
        duration = end_sec - start_sec
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output_path = Path(output)
            if output_path.exists() and output_path.stat().st_size >= 1024:
                return ClipResult(source, output, start_sec, end_sec, duration, method, True, None, output_path.stat().st_size)
            else:
                return self._error_result(source, output, start_sec, end_sec, "Output file too small or missing after FFmpeg run", method)
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg command failed with code {e.returncode}. stderr: {e.stderr}"
            return self._error_result(source, output, start_sec, end_sec, error_msg, method)
        except Exception as e:
            return self._error_result(source, output, start_sec, end_sec, str(e), method)

    def _try_opencv(self, source: str, output: str, start_sec: float, end_sec: float) -> ClipResult:
        duration = end_sec - start_sec
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                return self._error_result(source, output, start_sec, end_sec, "OpenCV failed to open source video", "opencv")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps <= 0 or width <= 0 or height <= 0:
                cap.release()
                return self._error_result(source, output, start_sec, end_sec, "Invalid video properties", "opencv")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame

            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                current_frame += 1

            cap.release()
            writer.release()
            
            output_path = Path(output)
            if output_path.exists() and output_path.stat().st_size > 0:
                return ClipResult(source, output, start_sec, end_sec, duration, "opencv", True, None, output_path.stat().st_size)
            else:
                return self._error_result(source, output, start_sec, end_sec, "Output file empty or missing after OpenCV run", "opencv")

        except Exception as e:
            return self._error_result(source, output, start_sec, end_sec, str(e), "opencv")

    def _error_result(self, source: str, output: str, start_sec: float, end_sec: float, error: str, method: str = "unknown") -> ClipResult:
        return ClipResult(source, output, start_sec, end_sec, end_sec - start_sec, method, False, error, 0)

    def cut_clips_batch(self, source_path: str, clips: list) -> List[ClipResult]:
        results = []
        for i, clip in enumerate(clips):
            if isinstance(clip, dict):
                start = float(clip.get('start', 0))
                end = float(clip.get('end', 0))
            else:
                start = float(getattr(clip, 'start', 0))
                end = float(getattr(clip, 'end', 0))
                
            filename = f"clip_{i+1:03d}_{start:.1f}s_{end:.1f}s.mp4"
            result = self.cut_clip(source_path, start, end, filename)
            results.append(result)
        return results

if __name__ == '__main__':
    cutter = ClipCutter()
    print(f"FFmpeg available: {cutter.ffmpeg_available}")
    
    test_video_path = Path("C:/SIH/SIH-2026/test_video.mp4")
    if test_video_path.exists():
        print(f"Found test video at {test_video_path}. Cutting 5-second clip...")
        result = cutter.cut_clip(str(test_video_path), 0.0, 5.0)
        print(f"Result: {result.to_dict()}")
    else:
        print(f"Test video not found at {test_video_path}. Skipping test cut.")
