# The per-video workflow. The review gate is a real paused state
# (waitForTaskToken), not a convention — nothing uncertain publishes on its own.
#
#   TranscribeDiarize (Batch, sync)
#     -> Attribute (Lambda: Claude vision + attribution)
#       -> auto_ok ————————————————————-> Publish
#       -> needs_review -> AwaitReview (paused, editor callback) -> Publish
#       -> failed ——————————————————————> MarkFailed

resource "aws_iam_role" "sfn" {
  name = "${var.project}-sfn"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "states.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "sfn" {
  name = "pipeline"
  role = aws_iam_role.sfn.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["batch:SubmitJob", "batch:DescribeJobs", "batch:TerminateJob"]
        Resource = "*"
      },
      {
        # required for the .sync Batch integration
        Effect = "Allow"
        Action = ["events:PutTargets", "events:PutRule", "events:DescribeRule"]
        Resource = [
          "arn:aws:events:${var.region}:${data.aws_caller_identity.current.account_id}:rule/StepFunctionsGetEventsForBatchJobsRule"
        ]
      },
      {
        Effect = "Allow"
        Action = ["lambda:InvokeFunction"]
        Resource = [
          aws_lambda_function.fn["attribute"].arn,
          aws_lambda_function.fn["notify_review"].arn,
          aws_lambda_function.fn["publish"].arn,
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["dynamodb:UpdateItem"]
        Resource = aws_dynamodb_table.jobs.arn
      },
    ]
  })
}

resource "aws_sfn_state_machine" "pipeline" {
  name     = "${var.project}-pipeline"
  role_arn = aws_iam_role.sfn.arn

  definition = jsonencode({
    Comment = "Video -> transcript with named speakers, gated by human review"
    StartAt = "TranscribeDiarize"
    States = {

      TranscribeDiarize = {
        Type     = "Task"
        Resource = "arn:aws:states:::batch:submitJob.sync"
        Parameters = {
          JobName       = "transcribe"
          JobQueue      = aws_batch_job_queue.gpu.arn
          JobDefinition = aws_batch_job_definition.transcribe.arn
          ContainerOverrides = {
            Environment = [
              { Name = "VIDEO_KEY", "Value.$" = "$.video_key" },
              { Name = "VIDEO_ID", "Value.$" = "$.video_id" },
              { Name = "WORK_PREFIX", "Value.$" = "$.work_prefix" },
            ]
          }
        }
        ResultPath = null # keep the original input flowing
        Retry = [{
          ErrorEquals     = ["States.ALL"]
          IntervalSeconds = 60
          MaxAttempts     = 2
          BackoffRate     = 2
        }]
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.error"
          Next        = "MarkFailed"
        }]
        Next = "Attribute"
      }

      Attribute = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.fn["attribute"].function_name
          "Payload.$"  = "$"
        }
        ResultSelector = { "status.$" = "$.Payload.status" }
        ResultPath     = "$.attr"
        Retry = [{
          ErrorEquals     = ["Lambda.TooManyRequestsException", "Lambda.ServiceException"]
          IntervalSeconds = 10
          MaxAttempts     = 3
          BackoffRate     = 2
        }]
        Catch = [{
          ErrorEquals = ["States.ALL"]
          ResultPath  = "$.error"
          Next        = "MarkFailed"
        }]
        Next = "StatusChoice"
      }

      StatusChoice = {
        Type = "Choice"
        Choices = [
          {
            Variable     = "$.attr.status"
            StringEquals = "auto_ok"
            Next         = "Publish"
          },
          {
            Variable     = "$.attr.status"
            StringEquals = "failed"
            Next         = "MarkFailed"
          },
        ]
        Default = "AwaitReview"
      }

      AwaitReview = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke.waitForTaskToken"
        Parameters = {
          FunctionName = aws_lambda_function.fn["notify_review"].function_name
          Payload = {
            "video_id.$"   = "$.video_id"
            "bucket.$"     = "$.bucket"
            "out_prefix.$" = "$.out_prefix"
            "task_token.$" = "$$.Task.Token"
          }
        }
        TimeoutSeconds = var.review_timeout_hours * 3600
        ResultPath     = "$.review"
        Catch = [{
          ErrorEquals = ["States.ALL"] # rejected or review timed out
          ResultPath  = "$.error"
          Next        = "MarkFailed"
        }]
        Next = "Publish"
      }

      Publish = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = aws_lambda_function.fn["publish"].function_name
          "Payload.$"  = "$"
        }
        ResultPath = "$.published"
        End        = true
      }

      MarkFailed = {
        Type     = "Task"
        Resource = "arn:aws:states:::dynamodb:updateItem"
        Parameters = {
          TableName = aws_dynamodb_table.jobs.name
          Key = {
            video_id = { "S.$" = "$.video_id" }
          }
          UpdateExpression         = "SET #s = :s"
          ExpressionAttributeNames = { "#s" = "status" }
          ExpressionAttributeValues = {
            ":s" = { S = "failed" }
          }
        }
        Next = "FailState"
      }

      FailState = {
        Type  = "Fail"
        Error = "PipelineFailed"
      }
    }
  })
}
