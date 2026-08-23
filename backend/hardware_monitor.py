"""
Hardware Monitoring Module for TraceVision CCTV Surveillance System (SIH-2026).

This module provides hardware monitoring capabilities, specifically tracking CPU, RAM,
and GPU VRAM usage. A critical constraint for the local testing environment of this
hackathon project is a strict 6 GB (6144 MB) VRAM ceiling. This module helps enforce
this constraint by monitoring VRAM allocation, logging warnings when nearing the limit,
and providing utility methods to check if operations can proceed safely.
"""

import logging
import psutil
import json
from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime, timezone

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

logger = logging.getLogger(__name__)

@dataclass
class ResourceSnapshot:
    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    vram_used_mb: Optional[float]
    vram_total_mb: Optional[float]
    vram_free_mb: Optional[float]
    is_gpu_available: bool
    is_within_vram_ceiling: bool
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "cpu_percent": self.cpu_percent,
            "ram_used_mb": self.ram_used_mb,
            "ram_total_mb": self.ram_total_mb,
            "vram_used_mb": self.vram_used_mb,
            "vram_total_mb": self.vram_total_mb,
            "vram_free_mb": self.vram_free_mb,
            "is_gpu_available": self.is_gpu_available,
            "is_within_vram_ceiling": self.is_within_vram_ceiling,
            "timestamp": self.timestamp
        }


class HardwareMonitor:
    def __init__(self, vram_ceiling_mb: float = 6144.0, warning_threshold: float = 0.80):
        self.vram_ceiling_mb = vram_ceiling_mb
        self.warning_threshold = warning_threshold
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._has_gpu = pynvml.nvmlDeviceGetCount() > 0
            except Exception as e:
                logger.warning(f"Failed to initialize pynvml: {e}")
                self._has_gpu = False
        else:
            self._has_gpu = False

    def check_resources(self) -> ResourceSnapshot:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used_mb = mem.used / (1024 * 1024)
        ram_total_mb = mem.total / (1024 * 1024)

        vram_used_mb = None
        vram_total_mb = None
        vram_free_mb = None
        is_within_vram_ceiling = True

        if self._has_gpu:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_used_mb = info.used / (1024 * 1024)
                vram_total_mb = info.total / (1024 * 1024)
                vram_free_mb = info.free / (1024 * 1024)
                
                # Check against ceiling rather than physical limit
                is_within_vram_ceiling = vram_used_mb <= self.vram_ceiling_mb
            except Exception as e:
                logger.error(f"Error querying GPU memory: {e}")

        return ResourceSnapshot(
            cpu_percent=cpu_percent,
            ram_used_mb=ram_used_mb,
            ram_total_mb=ram_total_mb,
            vram_used_mb=vram_used_mb,
            vram_total_mb=vram_total_mb,
            vram_free_mb=vram_free_mb,
            is_gpu_available=self._has_gpu,
            is_within_vram_ceiling=is_within_vram_ceiling,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def is_safe_to_proceed(self, required_mb: float = 0) -> bool:
        if not self._has_gpu:
            return True
        snapshot = self.check_resources()
        if snapshot.vram_used_mb is None:
            return True
        return (snapshot.vram_used_mb + required_mb) <= self.vram_ceiling_mb

    def get_status_headers(self) -> dict:
        snapshot = self.check_resources()
        headers = {
            "X-RAM-Used-MB": f"{snapshot.ram_used_mb:.2f}",
            "X-CPU-Percent": f"{snapshot.cpu_percent:.2f}",
        }
        if snapshot.is_gpu_available and snapshot.vram_used_mb is not None:
            headers["X-VRAM-Used-MB"] = f"{snapshot.vram_used_mb:.2f}"
            headers["X-VRAM-Free-MB"] = f"{snapshot.vram_free_mb:.2f}"
            headers["X-VRAM-Ceiling-MB"] = f"{self.vram_ceiling_mb:.2f}"
            
        return headers

    def log_warning_if_high(self):
        if not self._has_gpu:
            return
        snapshot = self.check_resources()
        if snapshot.vram_used_mb is not None:
            usage_ratio = snapshot.vram_used_mb / self.vram_ceiling_mb
            if usage_ratio > self.warning_threshold:
                logger.warning(
                    f"HIGH VRAM USAGE WARNING: {snapshot.vram_used_mb:.2f} MB used, "
                    f"which is {usage_ratio * 100:.1f}% of the {self.vram_ceiling_mb} MB ceiling."
                )

def add_hardware_headers_middleware(app, monitor: HardwareMonitor):
    """
    Adds a FastAPI middleware to append hardware resource headers to every response.
    Requires FastAPI application object.
    """
    @app.middleware("http")
    async def hardware_headers_middleware(request, call_next):
        response = await call_next(request)
        headers = monitor.get_status_headers()
        for key, value in headers.items():
            response.headers[key] = value
        return response

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor = HardwareMonitor()
    print("=== Hardware Resource Snapshot ===")
    snapshot = monitor.check_resources()
    print(json.dumps(snapshot.to_dict(), indent=2))
    
    print("\nSafety Check:")
    safe = monitor.is_safe_to_proceed()
    print(f"Is safe to proceed (0MB required)? {'YES' if safe else 'NO'}")
    
    monitor.log_warning_if_high()
