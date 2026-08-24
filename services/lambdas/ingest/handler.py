"""Ingest Lambda: SQS (S3-created events via EventBridge) -> Step Functions.

One execution per uploaded video. Idempotent: a re-upload of the same key
starts a fresh execution named with the object version/etag so reprocessing
is explicit and traceable.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import boto3

sfn = boto3.client("stepfunctions")
ddb = boto3.resource("dynamodb").Table(os.environ["JOBS_TABLE"])
s3 = boto3.client("s3")

STATE_MACHINE_ARN = os.environ["STATE_MACHINE_ARN"]
VIDEO_EXTS = (".mp4", ".mov", ".mxf", ".mkv", ".ts")


def _shotlist_key(bucket: str, video_id: str) -> str | None:
    key = f"ingest/shotlists/{video_id}.txt"
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return key
    except s3.exceptions.ClientError:
        return None


def handler(event, _context):
    started = []
    for record in event.get("Records", []):
        body = json.loads(record["body"])          # EventBridge envelope
        detail = body.get("detail", body)
        bucket = detail["bucket"]["name"]
        key = detail["object"]["key"]
        if not key.lower().endswith(VIDEO_EXTS):
            continue

        video_id = re.sub(r"[^A-Za-z0-9_-]", "_", key.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        etag = detail["object"].get("etag", "0")[:12]

        ddb.put_item(Item={
            "video_id": video_id,
            "status": "processing",
            "video_key": key,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

        execution = sfn.start_execution(
            stateMachineArn=STATE_MACHINE_ARN,
            name=f"{video_id}-{etag}"[:80],
            input=json.dumps({
                "video_id": video_id,
                "bucket": bucket,
                "video_key": key,
                "shotlist_key": _shotlist_key(bucket, video_id),
                "work_prefix": f"work/{video_id}",
                "out_prefix": f"out/{video_id}",
            }),
        )
        started.append(execution["executionArn"])
    return {"started": started}
