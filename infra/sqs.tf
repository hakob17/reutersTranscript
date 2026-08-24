# Ingest queue between S3/EventBridge and the ingest Lambda.
# Absorbs wire-service bursts; DLQ catches poison events.

resource "aws_sqs_queue" "ingest_dlq" {
  name                      = "${var.project}-ingest-dlq"
  message_retention_seconds = 1209600 # 14 days
}

resource "aws_sqs_queue" "ingest" {
  name                       = "${var.project}-ingest"
  visibility_timeout_seconds = 120 # > ingest lambda timeout
  message_retention_seconds  = 86400

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.ingest_dlq.arn
    maxReceiveCount     = 4
  })
}

resource "aws_cloudwatch_event_rule" "video_uploaded" {
  name = "${var.project}-video-uploaded"

  event_pattern = jsonencode({
    source      = ["aws.s3"]
    detail-type = ["Object Created"]
    detail = {
      bucket = { name = [aws_s3_bucket.data.bucket] }
      object = { key = [{ prefix = "ingest/videos/" }] }
    }
  })
}

resource "aws_cloudwatch_event_target" "video_uploaded_to_sqs" {
  rule = aws_cloudwatch_event_rule.video_uploaded.name
  arn  = aws_sqs_queue.ingest.arn
}

resource "aws_sqs_queue_policy" "allow_eventbridge" {
  queue_url = aws_sqs_queue.ingest.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "events.amazonaws.com" }
      Action    = "sqs:SendMessage"
      Resource  = aws_sqs_queue.ingest.arn
      Condition = {
        ArnEquals = { "aws:SourceArn" = aws_cloudwatch_event_rule.video_uploaded.arn }
      }
    }]
  })
}
