"""Generate tiny REAL audio files (via PyAV) so tag-writing is tested on genuine containers."""

from __future__ import annotations

import av
import numpy as np
from pathlib import Path

SPECS = {".mp3": ("libmp3lame", "mp3", 44100), ".flac": ("flac", "flac", 44100),
         ".ogg": ("libvorbis", "ogg", 44100), ".opus": ("libopus", "ogg", 48000),
         ".m4a": ("aac", "ipod", 44100)}

def make_audio(path, seconds: float = 2.0) -> bool:
    """Write a sine tone to *path* (format chosen by suffix). False if this PyAV lacks the encoder."""
    path = Path(path); codec, container, rate = SPECS[path.suffix.lower()]
    candidates = [codec] + (["vorbis"] if codec == "libvorbis" else [])
    last = None
    for name in candidates:
        try:
            out = av.open(str(path), "w", format=container)
            st = out.add_stream(name, rate=rate)
            if name == "vorbis":
                st.codec_context.options = {"strict": "-2"}
            st.layout = "mono"
            fmt = st.codec_context.format.name
            size = st.codec_context.frame_size or 1024
            t = np.arange(int(rate * seconds)) / rate
            sig = (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32").reshape(1, -1)
            for i in range(0, sig.shape[1] - size + 1, size):
                chunk = sig[:, i:i + size]
                if fmt.startswith("s16"): chunk = (chunk * 32767).astype("int16")
                if fmt.startswith("s32"): chunk = (chunk * 2147483647).astype("int32")
                fr = av.AudioFrame.from_ndarray(chunk, format=fmt, layout="mono"); fr.sample_rate = rate; fr.pts = i
                for p in st.encode(fr): out.mux(p)
            for p in st.encode(None): out.mux(p)
            out.close(); return True
        except Exception as exc:
            last = exc
            try: out.close()
            except Exception: pass
            path.unlink(missing_ok=True)
    return False
