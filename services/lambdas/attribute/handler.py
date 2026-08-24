"""Attribute Lambda: chyron reading (Claude vision) + speaker attribution.

Input (from Step Functions, after the Batch GPU task):
  {video_id, bucket, work_prefix, out_prefix, shotlist_key}

Reads segments.json + crops from S3, calls Claude twice (read crops, then
attribute), writes .vtt/.json/.labeled.txt to out_prefix, updates DynamoDB,
and returns {"status": "auto_ok" | "needs_review" | "failed"}.

Bundled with the speaker_attribution package (no cv2 needed — detection
already happened on the GPU task).
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import boto3

from speaker_attribution.attribute import attribute_speakers
from speaker_attribution.chyron import merge_chyrons, read_chyron_crops
from speaker_attribution.models import Segment
from speaker_attribution.outputs import write_json, write_labeled_txt, write_vtt

s3 = boto3.client("s3")
ddb = boto3.resource("dynamodb").Table(os.environ["JOBS_TABLE"])


def _anthropic_client() -> anthropic.Anthropic:
    sm = boto3.client("secretsmanager")
    key = sm.get_secret_value(
        SecretId=os.environ["ANTHROPIC_SECRET_ARN"])["SecretString"]
    return anthropic.Anthropic(api_key=key)


def _read_json(bucket: str, key: str):
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def handler(event, _context):
    video_id = event["video_id"]
    bucket = event["bucket"]
    work = event["work_prefix"].rstrip("/")
    out = event["out_prefix"].rstrip("/")
    client = _anthropic_client()

    segments = [Segment(**s) for s in _read_json(bucket, f"{work}/segments.json")]

    crops = []
    for c in _read_json(bucket, f"{work}/crops.json"):
        body = s3.get_object(Bucket=bucket, Key=c["key"])["Body"].read()
        crops.append({"t": c["t"], "label": c["label"], "jpeg": body})
    sightings, chyron_warnings = read_chyron_crops(crops, client)

    shotlist = ""
    if event.get("shotlist_key"):
        shotlist = s3.get_object(
            Bucket=bucket, Key=event["shotlist_key"])["Body"].read().decode()

    result = attribute_speakers(
        video_id=video_id,
        segments=segments,
        shotlist=shotlist,
        byline=event.get("byline", ""),
        client=client,
    )
    result = merge_chyrons(result, sightings, chyron_warnings)

    with tempfile.TemporaryDirectory() as td:
        for writer, suffix in ((write_vtt, ".vtt"), (write_json, ".json"),
                               (write_labeled_txt, ".labeled.txt")):
            local = Path(td) / f"{video_id}{suffix}"
            writer(result, local)
            s3.upload_file(str(local), bucket, f"{out}/{video_id}{suffix}")

    ddb.update_item(
        Key={"video_id": video_id},
        UpdateExpression="SET #s = :s, out_prefix = :o, warnings = :w, updated_at = :t",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": result.status.value,
            ":o": out,
            ":w": result.warnings[:20],
            ":t": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {"status": result.status.value, "warnings": len(result.warnings)}
