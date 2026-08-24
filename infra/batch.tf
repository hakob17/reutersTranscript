# GPU transcription workers: AWS Batch on EC2 (g6/g5), scale-to-zero.
# Spot by default — jobs are minutes-long and idempotent, so interruption
# just means an automatic retry.

resource "aws_ecr_repository" "gpu_task" {
  name                 = "${var.project}-gpu"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# --- IAM ------------------------------------------------------------------

resource "aws_iam_role" "batch_service" {
  name = "${var.project}-batch-service"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "batch.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_iam_role" "batch_instance" {
  name = "${var.project}-batch-instance"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_instance_ecs" {
  role       = aws_iam_role.batch_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch_instance" {
  name = "${var.project}-batch-instance"
  role = aws_iam_role.batch_instance.name
}

# Role the job container runs as: S3 data prefixes + HF token secret.
resource "aws_iam_role" "batch_job" {
  name = "${var.project}-batch-job"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "batch_job" {
  name = "data-access"
  role = aws_iam_role.batch_job.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = "${aws_s3_bucket.data.arn}/ingest/*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:GetObject"]
        Resource = "${aws_s3_bucket.data.arn}/work/*"
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = aws_secretsmanager_secret.hf_token.arn
      },
    ]
  })
}

# --- Compute --------------------------------------------------------------

resource "aws_batch_compute_environment" "gpu" {
  compute_environment_name = "${var.project}-gpu"
  type                     = "MANAGED"
  service_role             = aws_iam_role.batch_service.arn

  compute_resources {
    type                = var.use_spot ? "SPOT" : "EC2"
    allocation_strategy = var.use_spot ? "SPOT_PRICE_CAPACITY_OPTIMIZED" : "BEST_FIT_PROGRESSIVE"
    instance_type       = var.gpu_instance_types
    min_vcpus           = 0 # scale to zero when the queue is empty
    max_vcpus           = var.batch_max_vcpus
    instance_role       = aws_iam_instance_profile.batch_instance.arn
    security_group_ids  = [aws_security_group.batch.id]
    subnets             = aws_subnet.public[*].id
  }

  depends_on = [aws_iam_role_policy_attachment.batch_service]
}

resource "aws_batch_job_queue" "gpu" {
  name     = "${var.project}-gpu"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu.arn
  }
}

resource "aws_cloudwatch_log_group" "gpu_task" {
  name              = "/aws/batch/${var.project}-gpu"
  retention_in_days = 30
}

resource "aws_batch_job_definition" "transcribe" {
  name = "${var.project}-transcribe"
  type = "container"

  # Spot interruptions and transient HF/S3 failures: retry, but not forever.
  retry_strategy {
    attempts = 3
  }

  timeout {
    attempt_duration_seconds = 3600
  }

  container_properties = jsonencode({
    image      = "${aws_ecr_repository.gpu_task.repository_url}:${var.gpu_image_tag}"
    jobRoleArn = aws_iam_role.batch_job.arn

    resourceRequirements = [
      { type = "VCPU", value = "4" },
      { type = "MEMORY", value = "15000" },
      { type = "GPU", value = "1" },
    ]

    environment = [
      { name = "DATA_BUCKET", value = aws_s3_bucket.data.bucket },
      { name = "HF_TOKEN_SECRET_ARN", value = aws_secretsmanager_secret.hf_token.arn },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group" = aws_cloudwatch_log_group.gpu_task.name
      }
    }
  })
}
