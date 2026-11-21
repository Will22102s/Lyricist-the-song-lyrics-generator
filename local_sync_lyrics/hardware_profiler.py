"""Assess the machine once at start-up and recommend a Whisper configuration.

Everything is best-effort: any probe that fails simply leaves its field empty and
the recommendation degrades gracefully to a conservative CPU model.

GPU use requires *all* of: a CUDA device visible to CTranslate2, working CUDA 12
runtime libraries (cuBLAS) and a supported compute type. We verify the libraries
up-front by trying to load them, so a missing cuBLAS is reported immediately
(and the transcriber can pick CPU) instead of crashing mid-batch.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
_DLL_DIR_HANDLES: list = []  # keep os.add_dll_directory handles alive


@dataclass
class Recommendation:
    model: str
    device: str          # "cuda" | "cpu"
    compute_type: str    # e.g. float16, int8_float16, int8
    reason: str


@dataclass
class HardwareProfile:
    cpu_name: str = ""
    physical_cores: int = 1
    logical_cores: int = 1
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    gpu_name: str = ""
    gpu_vram_gb: Optional[float] = None
    cuda_devices: int = 0
    cuda_libs_ok: bool = False
    cuda_compute_types: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    recommendation: Recommendation = field(
        default_factory=lambda: Recommendation("tiny", "cpu", "int8", "default")
    )
    cpu_recommendation: Recommendation = field(
        default_factory=lambda: Recommendation("tiny", "cpu", "int8", "default")
    )

    @property
    def cuda_usable(self) -> bool:
        return self.cuda_devices > 0 and self.cuda_libs_ok

    def badge(self) -> str:
        """Short one-liner for the UI header."""
        if self.cuda_usable:
            vram = f" · {self.gpu_vram_gb:.0f} GB" if self.gpu_vram_gb else ""
            return f"GPU: {self.gpu_name or 'CUDA device'}{vram}"
        return f"CPU: {self.physical_cores} cores · {self.ram_total_gb:.0f} GB RAM"

    def summary_lines(self) -> list[str]:
        lines = [
            f"CPU: {self.cpu_name or platform.machine()} "
            f"({self.physical_cores} cores / {self.logical_cores} threads)",
            f"RAM: {self.ram_total_gb:.1f} GB total, {self.ram_available_gb:.1f} GB free",
        ]
        if self.cuda_devices:
            vram = f", {self.gpu_vram_gb:.1f} GB VRAM" if self.gpu_vram_gb else ""
            lines.append(f"GPU: {self.gpu_name or 'CUDA device'}{vram} "
                         f"(CUDA libs {'OK' if self.cuda_libs_ok else 'MISSING'})")
        elif self.gpu_name:
            lines.append(f"GPU: {self.gpu_name} (not usable by CTranslate2)")
        else:
            lines.append("GPU: none detected")
        rec = self.recommendation
        lines.append(f"Recommended: {rec.model} on {rec.device} ({rec.compute_type}) - {rec.reason}")
        lines.extend(f"Note: {n}" for n in self.notes)
        return lines


# ------------------------------------------------------------------- probes
def _ram_gb() -> tuple[float, float]:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return mem.total / 2**30, mem.available / 2**30
    except Exception:
        pass
    try:
        if sys.platform.startswith("linux"):
            info = {}
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    key, _, rest = line.partition(":")
                    info[key] = int(rest.split()[0]) * 1024
            return info["MemTotal"] / 2**30, info.get("MemAvailable", info["MemFree"]) / 2**30
        if sys.platform == "win32":
            class MEMSTATUS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = MEMSTATUS()
            status.dwLength = ctypes.sizeof(MEMSTATUS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
            return status.ullTotalPhys / 2**30, status.ullAvailPhys / 2**30
    except Exception as exc:
        logger.debug("RAM probe failed: %s", exc)
    return 0.0, 0.0


def _cpu_info() -> tuple[str, int, int]:
    logical = os.cpu_count() or 1
    physical = logical
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or logical
    except Exception:
        pass
    name = platform.processor() or ""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return name, max(1, physical), max(1, logical)


def _nvidia_smi() -> tuple[str, Optional[float]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return "", None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6, creationflags=_NO_WINDOW,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return "", None
        name, _, mem = out.stdout.strip().splitlines()[0].partition(",")
        return name.strip(), float(mem.strip()) / 1024.0
    except (OSError, ValueError, subprocess.SubprocessError):
        return "", None


def _site_dirs() -> list[str]:
    dirs = [p for p in sys.path if p and os.path.isdir(p)]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.insert(0, meipass)
    return dirs


def register_pip_cuda_libraries() -> list[str]:
    """Make CUDA libs installed via ``pip install nvidia-cublas-cu12`` loadable.

    Windows: add ``site-packages/nvidia/*/bin`` to the DLL search path.
    Linux  : pre-load the shared objects (LD_LIBRARY_PATH cannot be changed
             after the interpreter has started).
    Returns the directories that were registered.
    """
    found: list[str] = []
    for base in _site_dirs():
        pattern = os.path.join(base, "nvidia", "*", "bin" if os.name == "nt" else "lib")
        found.extend(d for d in glob.glob(pattern) if os.path.isdir(d))
    for directory in found:
        try:
            if os.name == "nt":
                _DLL_DIR_HANDLES.append(os.add_dll_directory(directory))  # type: ignore[attr-defined]
                os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
            else:
                for lib in sorted(glob.glob(os.path.join(directory, "lib*.so*"))):
                    if any(k in os.path.basename(lib) for k in ("cublas", "cudnn", "cudart", "nvrtc")):
                        try:
                            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            pass
        except OSError as exc:
            logger.debug("could not register %s: %s", directory, exc)
    return found


def cuda_libraries_ok() -> tuple[bool, str]:
    """Try to load cuBLAS (CUDA 12) - the library CTranslate2 needs on GPU."""
    register_pip_cuda_libraries()
    names = ("cublas64_12.dll",) if os.name == "nt" else ("libcublas.so.12", "libcublas.so")
    errors = []
    for name in names:
        try:
            ctypes.CDLL(name)
            return True, name
        except OSError as exc:
            errors.append(str(exc))
    return False, "cuBLAS 12 not found"


def _ctranslate2_cuda() -> tuple[int, list[str]]:
    try:
        import ctranslate2

        count = int(ctranslate2.get_cuda_device_count())
        types = sorted(ctranslate2.get_supported_compute_types("cuda")) if count else []
        return count, types
    except Exception as exc:  # missing wheel, driver problems...
        logger.debug("ctranslate2 CUDA probe failed: %s", exc)
        return 0, []


# ----------------------------------------------------------- recommendation
def _gpu_compute_type(types: list[str], vram_gb: Optional[float]) -> str:
    tight = vram_gb is not None and vram_gb < 6
    order = ["int8_float16", "float16"] if tight else ["float16", "int8_float16"]
    for candidate in order + ["int8", "float32"]:
        if candidate in types:
            return candidate
    return "float16"


def recommend_cpu(cores: int, ram_gb: float) -> Recommendation:
    if ram_gb and ram_gb < 3.5 or cores <= 2:
        return Recommendation("tiny", "cpu", "int8", "low-spec CPU: fastest model")
    if (ram_gb and ram_gb < 8) or cores <= 4:
        return Recommendation("base", "cpu", "int8", "modest CPU: balanced model")
    if cores >= 8 and ram_gb >= 12:
        return Recommendation("small", "cpu", "int8", "strong CPU: best CPU-friendly model")
    return Recommendation("base", "cpu", "int8", "mid-range CPU: balanced model")


def recommend(profile: HardwareProfile) -> Recommendation:
    cpu = recommend_cpu(profile.physical_cores, profile.ram_total_gb)
    if not profile.cuda_usable:
        return cpu
    vram = profile.gpu_vram_gb
    ctype = _gpu_compute_type(profile.cuda_compute_types, vram)
    if vram is None or vram >= 6:
        return Recommendation("large-v3", "cuda", ctype, "GPU with plenty of VRAM: best accuracy")
    if vram >= 3.5:
        return Recommendation("large-v3-turbo", "cuda", ctype, "GPU with limited VRAM: fast & accurate")
    if vram >= 2:
        return Recommendation("small", "cuda", ctype, "small GPU: lightweight model")
    return cpu


# ------------------------------------------------------------------- public
_cached: Optional[HardwareProfile] = None
_lock = threading.Lock()


def get_profile(refresh: bool = False) -> HardwareProfile:
    """Probe the machine (cached). Safe to call from any thread."""
    global _cached
    with _lock:
        if _cached is not None and not refresh:
            return _cached
        profile = HardwareProfile()
        try:
            profile.cpu_name, profile.physical_cores, profile.logical_cores = _cpu_info()
            profile.ram_total_gb, profile.ram_available_gb = _ram_gb()
            profile.gpu_name, profile.gpu_vram_gb = _nvidia_smi()
            profile.cuda_devices, profile.cuda_compute_types = _ctranslate2_cuda()
            if profile.cuda_devices:
                profile.cuda_libs_ok, detail = cuda_libraries_ok()
                if not profile.cuda_libs_ok:
                    profile.notes.append(
                        "CUDA GPU found but cuBLAS 12 could not be loaded - using CPU. "
                        "Install the NVIDIA CUDA 12 runtime or `pip install nvidia-cublas-cu12`."
                    )
            elif profile.gpu_name:
                profile.notes.append(
                    "NVIDIA GPU present but CTranslate2 reports no CUDA device (driver/wheel issue)."
                )
            if profile.ram_total_gb and profile.ram_total_gb < 4:
                profile.notes.append("Less than 4 GB RAM: only the tiny/base models are advisable.")
        except Exception as exc:  # absolutely never fail start-up on a probe
            logger.exception("hardware probe failed")
            profile.notes.append(f"Hardware probe incomplete: {exc.__class__.__name__}")
        profile.cpu_recommendation = recommend_cpu(profile.physical_cores, profile.ram_total_gb)
        profile.recommendation = recommend(profile)
        _cached = profile
        return profile
