# Per-video job state: processing -> auto_ok|needs_review -> awaiting_review
# -> approved|rejected -> published. Also holds the Step Functions task token
# while a video waits for an editor.

resource "aws_dynamodb_table" "jobs" {
  name         = "${var.project}-jobs"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "video_id"

  attribute {
    name = "video_id"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }
}
