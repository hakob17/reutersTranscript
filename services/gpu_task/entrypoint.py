"""AWS Batch GPU task: transcribe + diarize + detect chyron frames.

Runs the compute-heavy, video-local stages and hands everything else to
Lambdas via S3. One job per video, driven entirely by environment variables
(set by the Step Functions ContainerOverrides):

  DATA_BUCKET   bucket holding work/ and out/ prefixes
  VIDEO_KEY     s3 key of the source video (ingest/videos/<id>.mp4), OR
  VIDEO_URL     HLS/HTTP stream URL — read directly by ffmpeg/OpenCV, no
                MP4 or S3 copy of the source needed (wire-CDN friendly)
  VIDEO_ID      stem used for all derived keys
  WORK_PREFIX   where to write intermediates (work/<id>/)
  HF_TOKEN_SECRET_ARN  Secrets Manager ARN with the HuggingFace token
  WHISPER_MODEL / DEVICE / COMPUTE_TYPE  optional overrides

Writes to s3://DATA_BUCKET/WORK_PREFIX:
  segments.json   [{start, end, speaker, text, language}, ...]
  crops/NNN.jpg   candidate lower-third crops (chyron detection)
  crops.json      [{key, label, t}, ...] manifest for the vision Lambda
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent))

from speaker_attribution.chyron import _find_chyron_frames  # noqa: E402
from speaker_attribution.transcribe import extract_audio, transcribe_and_diarize  # noqa: E402


def hf_token(secret_arn: str) -> str:
    sm = boto3.client("secretsmanager")
    return sm.get_secret_value(SecretId=secret_arn)["SecretString"]


def main() -> int:
    bucket = os.environ["DATA_BUCKET"]
    video_key = os.environ.get("VIDEO_KEY", "")
    video_url = os.environ.get("VIDEO_URL", "")
    video_id = os.environ["VIDEO_ID"]
    work_prefix = os.environ["WORK_PREFIX"].rstrip("/")
    token = hf_token(os.environ["HF_TOKEN_SECRET_ARN"])

    s3 = boto3.client("s3")
    with tempfile.TemporaryDirectory() as td:
        if video_url:
            # HLS/HTTP source: ffmpeg and OpenCV read the stream directly.
            video: Path | str = video_url
            print(f"[{video_id}] reading stream {video_url}")
        else:
            video = Path(td) / Path(video_key).name
            print(f"[{video_id}] downloading s3://{bucket}/{video_key}")
            s3.download_file(bucket, video_key, str(video))

        print(f"[{video_id}] extracting audio")
        wav = extract_audio(video, Path(td) / f"{video_id}.wav")

        print(f"[{video_id}] transcribing + diarizing")
        segments = transcribe_and_diarize(
            wav,
            hf_token=token,
            model_name=os.environ.get("WHISPER_MODEL", "large-v3"),
            device=os.environ.get("DEVICE", "cuda"),
            compute_type=os.environ.get("COMPUTE_TYPE", "float16"),
        )
        print(f"[{video_id}] {len(segments)} segments, "
              f"{len({s.speaker for s in segments})} speakers")

        seg_payload = [
            {"start": s.start, "end": s.end, "speaker": s.speaker,
             "text": s.text, "language": s.language}
            for s in segments
        ]
        s3.put_object(
            Bucket=bucket, Key=f"{work_prefix}/segments.json",
            Body=json.dumps(seg_payload, ensure_ascii=False).encode(),
            ContentType="application/json",
        )

        print(f"[{video_id}] detecting lower-third frames (OpenCV)")
        events = _find_chyron_frames(video, segments)
        manifest = []
        for i, e in enumerate(events):
            key = f"{work_prefix}/crops/{i:03d}.jpg"
            s3.put_object(Bucket=bucket, Key=key, Body=e["jpeg"],
                          ContentType="image/jpeg")
            manifest.append({"key": key, "label": e["label"], "t": e["t"]})
        s3.put_object(
            Bucket=bucket, Key=f"{work_prefix}/crops.json",
            Body=json.dumps(manifest).encode(), ContentType="application/json",
        )
        print(f"[{video_id}] uploaded {len(manifest)} crops + segments.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
