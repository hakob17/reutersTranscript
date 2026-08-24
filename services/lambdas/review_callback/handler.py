"""Review-callback Lambda: an editor's approve/reject resumes the workflow.

Invoke with {"video_id": "...", "approve": true|false, "reason": "..."}.
Looks up the stored task token and calls SendTaskSuccess/Failure on the
paused Step Functions execution. This is the integration point for a future
review UI — the UI just invokes this function.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import boto3

ddb = boto3.resource("dynamodb").Table(os.environ["JOBS_TABLE"])
sfn = boto3.client("stepfunctions")


def handler(event, _context):
    video_id = event["video_id"]
    item = ddb.get_item(Key={"video_id": video_id}).get("Item")
    if not item or "task_token" not in item:
        return {"error": f"no pending review for {video_id}"}

    if event.get("approve"):
        sfn.send_task_success(taskToken=item["task_token"],
                              output='{"approved": true}')
        status = "approved"
    else:
        sfn.send_task_failure(taskToken=item["task_token"], error="Rejected",
                              cause=event.get("reason", "rejected by editor"))
        status = "rejected"

    ddb.update_item(
        Key={"video_id": video_id},
        UpdateExpression="SET #s = :s, updated_at = :t REMOVE task_token",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": status,
            ":t": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {"video_id": video_id, "result": status}
