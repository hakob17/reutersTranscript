"""Notify-review Lambda (invoked with .waitForTaskToken).

Step Functions pauses the workflow here for videos flagged needs_review.
Stores the task token in DynamoDB and emails reviewers via SNS; the
review_callback Lambda resumes the execution when an editor decides.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import boto3

ddb = boto3.resource("dynamodb").Table(os.environ["JOBS_TABLE"])
sns = boto3.client("sns")


def handler(event, _context):
    video_id = event["video_id"]
    token = event["task_token"]

    ddb.update_item(
        Key={"video_id": video_id},
        UpdateExpression="SET task_token = :tok, #s = :s, updated_at = :t",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":tok": token,
            ":s": "awaiting_review",
            ":t": datetime.now(timezone.utc).isoformat(),
        },
    )

    sns.publish(
        TopicArn=os.environ["REVIEW_TOPIC_ARN"],
        Subject=f"[transcript review] {video_id} needs speaker review",
        Message=json.dumps({
            "video_id": video_id,
            "outputs": f"s3://{event['bucket']}/{event['out_prefix']}/",
            "approve_with": (
                f"aws lambda invoke --function-name {os.environ['CALLBACK_FN']} "
                f"--payload '{{\"video_id\": \"{video_id}\", \"approve\": true}}' out.json"
            ),
        }, indent=2),
    )
    return {"notified": video_id}
