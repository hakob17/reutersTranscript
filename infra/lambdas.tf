# Five functions, zipped from build/<name>/ (run scripts/build_lambdas.sh
# first — it copies handlers, the speaker_attribution package, and pip deps
# into the build dirs).

locals {
  lambda_build_dir = "${path.module}/../build/lambdas"

  # Constructed as a string (not a resource reference) to break the
  # lambda -> state machine -> lambda dependency cycle.
  state_machine_arn = "arn:aws:states:${var.region}:${data.aws_caller_identity.current.account_id}:stateMachine:${var.project}-pipeline"

  lambdas = {
    ingest = {
      timeout = 60
      memory  = 256
      env = {
        JOBS_TABLE        = aws_dynamodb_table.jobs.name
        STATE_MACHINE_ARN = local.state_machine_arn
      }
    }
    attribute = {
      timeout = 600 # two Claude calls + S3 IO
      memory  = 1024
      env = {
        JOBS_TABLE           = aws_dynamodb_table.jobs.name
        ANTHROPIC_SECRET_ARN = aws_secretsmanager_secret.anthropic_api_key.arn
      }
    }
    notify_review = {
      timeout = 30
      memory  = 128
      env = {
        JOBS_TABLE       = aws_dynamodb_table.jobs.name
        REVIEW_TOPIC_ARN = aws_sns_topic.review.arn
        CALLBACK_FN      = "${var.project}-review_callback"
      }
    }
    review_callback = {
      timeout = 30
      memory  = 128
      env = {
        JOBS_TABLE = aws_dynamodb_table.jobs.name
      }
    }
    publish = {
      timeout = 60
      memory  = 256
      env = {
        JOBS_TABLE = aws_dynamodb_table.jobs.name
      }
    }
  }
}

# Per-function permissions beyond the shared DynamoDB access (least privilege
# per function). All Resource values are lists so the statement objects share
# one type — required for concat()/lookup() type unification.
locals {
  lambda_extra_statements = {
    ingest = [
      {
        Effect   = "Allow"
        Action   = ["states:StartExecution"]
        Resource = [local.state_machine_arn]
      },
      {
        Effect   = "Allow"
        Action   = ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"]
        Resource = [aws_sqs_queue.ingest.arn]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:ListBucket"]
        Resource = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/ingest/*"]
      },
    ]
    attribute = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.data.arn}/work/*", "${aws_s3_bucket.data.arn}/ingest/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = ["${aws_s3_bucket.data.arn}/out/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = [aws_secretsmanager_secret.anthropic_api_key.arn]
      },
    ]
    notify_review = [
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = [aws_sns_topic.review.arn]
      },
    ]
    review_callback = [
      {
        Effect   = "Allow"
        Action   = ["states:SendTaskSuccess", "states:SendTaskFailure"]
        Resource = ["*"] # task tokens are not resource-scopable
      },
    ]
    publish = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = ["${aws_s3_bucket.data.arn}/out/*"]
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = ["${aws_s3_bucket.data.arn}/published/*"]
      },
    ]
  }
}

data "archive_file" "lambda" {
  for_each = local.lambdas

  type        = "zip"
  source_dir  = "${local.lambda_build_dir}/${each.key}"
  output_path = "${local.lambda_build_dir}/${each.key}.zip"
}

resource "aws_iam_role" "lambda" {
  for_each = local.lambdas

  name = "${var.project}-lambda-${each.key}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  for_each = local.lambdas

  role       = aws_iam_role.lambda[each.key].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Per-function permissions, least-privilege by function.
resource "aws_iam_role_policy" "lambda" {
  for_each = local.lambdas

  name = "fn-access"
  role = aws_iam_role.lambda[each.key].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [
        {
          Effect   = "Allow"
          Action   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem"]
          Resource = [aws_dynamodb_table.jobs.arn]
        },
      ],
      lookup(local.lambda_extra_statements, each.key, []),
    )
  })
}

resource "aws_lambda_function" "fn" {
  for_each = local.lambdas

  function_name = "${var.project}-${each.key}"
  role          = aws_iam_role.lambda[each.key].arn
  runtime       = "python3.12"
  handler       = "handler.handler"
  timeout       = each.value.timeout
  memory_size   = each.value.memory

  filename         = data.archive_file.lambda[each.key].output_path
  source_code_hash = data.archive_file.lambda[each.key].output_base64sha256

  environment {
    variables = each.value.env
  }
}

resource "aws_lambda_event_source_mapping" "ingest_sqs" {
  event_source_arn                   = aws_sqs_queue.ingest.arn
  function_name                      = aws_lambda_function.fn["ingest"].arn
  batch_size                         = 5
  maximum_batching_window_in_seconds = 10
}
