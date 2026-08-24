"""Publish Lambda: copy approved outputs to the published/ prefix.

Runs for auto_ok videos and for needs_review videos after editor approval.
The published/ prefix is what the CMS / player site consumes.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import boto3

s3 = boto3.client("s3")
ddb = boto3.resource("dynamodb").Table(os.environ["JOBS_TABLE"])


def handler(event, _context):
    video_id = event["video_id"]
    bucket = event["bucket"]
    out = event["out_prefix"].rstrip("/")

    published = []
    for suffix in (".vtt", ".json", ".labeled.txt"):
        src = f"{out}/{video_id}{suffix}"
        dst = f"published/{video_id}/{video_id}{suffix}"
        s3.copy_object(Bucket=bucket, Key=dst,
                       CopySource={"Bucket": bucket, "Key": src})
        published.append(dst)

    ddb.update_item(
        Key={"video_id": video_id},
        UpdateExpression="SET #s = :s, published_prefix = :p, updated_at = :t",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": "published",
            ":p": f"published/{video_id}/",
            ":t": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {"video_id": video_id, "published": published}
