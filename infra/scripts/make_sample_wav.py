"""Generate a tiny silent WAV for smoke-testing /identify (stdlib only)."""
from __future__ import annotations
import struct, sys, wave

def make_wav(path: str, seconds: float = 4.0, sample_rate: int = 32000) -> None:
    n = int(seconds * sample_rate)
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))
    print(f"wrote {path} ({seconds}s @ {sample_rate}Hz)")

if __name__ == "__main__":
    make_wav(sys.argv[1] if len(sys.argv) > 1 else "sample.wav")
