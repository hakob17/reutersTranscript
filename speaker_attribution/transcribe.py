"""Stage 1-2: extract audio from video, run WhisperX ASR + diarization.

WhisperX is imported lazily so the attribution/output stages can be tested
on machines without GPU/torch installed.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .models import Segment

SAMPLE_RATE = 16_000


def extract_audio(video_path: Path | str, out_wav: Path | None = None) -> Path:
    """Extract mono 16 kHz WAV from any source ffmpeg understands — local
    files or HTTP(S)/HLS stream URLs (.m3u8) alike; no download step needed."""
    if out_wav is None:
        out_wav = Path(tempfile.mkstemp(suffix=".wav")[1])
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
        str(out_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_wav


def diarize_only(
    audio_path: Path,
    hf_token: str,
    device: str = "cuda",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[tuple[float, float, str]]:
    """Diarization without ASR — the caption-first path, where text and
    timings already come from the broadcaster's VTT. Returns speaker turns
    as (start, end, label)."""
    import whisperx  # lazy import
    from whisperx.diarize import DiarizationPipeline  # submodule not auto-imported

    audio = whisperx.load_audio(str(audio_path))
    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers
    df = diarize_model(audio, **kwargs)
    return [(float(r["start"]), float(r["end"]), str(r["speaker"]))
            for _, r in df.iterrows()]


def transcribe_and_diarize(
    audio_path: Path,
    hf_token: str,
    model_name: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[Segment]:
    """Run WhisperX: ASR -> word alignment -> pyannote diarization -> merged segments.

    On CPU use device="cpu", compute_type="int8", model_name="medium".
    """
    import whisperx  # lazy import

    audio = whisperx.load_audio(str(audio_path))

    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=16)
    language = result["language"]

    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)

    diarize_model = whisperx.diarize.DiarizationPipeline(token=hf_token, device=device)
    diarize_kwargs = {}
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers
    diarize_segments = diarize_model(audio, **diarize_kwargs)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    segments: list[Segment] = []
    for seg in result["segments"]:
        segments.append(
            Segment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                speaker=seg.get("speaker", "SPEAKER_UNKNOWN"),
                text=seg["text"].strip(),
                language=language,
            )
        )
    return segments
