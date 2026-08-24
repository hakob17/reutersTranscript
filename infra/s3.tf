# One data bucket, four prefixes:
#   ingest/videos/     uploads trigger the pipeline
#   ingest/shotlists/  optional <video_id>.txt shotlists
#   work/<id>/         intermediates (segments.json, crops) — auto-expired
#   out/<id>/          attribution outputs awaiting review
#   published/<id>/    approved outputs, consumed by the CMS/player

resource "aws_s3_bucket" "data" {
  bucket = "${var.project}-data-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "expire-work-intermediates"
    status = "Enabled"
    filter {
      prefix = "work/"
    }
    expiration {
      days = 14
    }
  }

  rule {
    id     = "expire-old-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# S3 -> EventBridge (object created)
resource "aws_s3_bucket_notification" "data" {
  bucket      = aws_s3_bucket.data.id
  eventbridge = true
}
