output "data_bucket" {
  value       = aws_s3_bucket.data.bucket
  description = "Upload videos to ingest/videos/, shotlists to ingest/shotlists/"
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.gpu_task.repository_url
  description = "Push the GPU worker image here (tag = var.gpu_image_tag)"
}

output "state_machine_arn" {
  value = aws_sfn_state_machine.pipeline.arn
}

output "review_callback_function" {
  value       = aws_lambda_function.fn["review_callback"].function_name
  description = "Invoke with {\"video_id\": ..., \"approve\": true|false} to resume a paused review"
}

output "anthropic_secret_arn" {
  value       = aws_secretsmanager_secret.anthropic_api_key.arn
  description = "Set the value with: aws secretsmanager put-secret-value"
}

output "hf_token_secret_arn" {
  value = aws_secretsmanager_secret.hf_token.arn
}

output "jobs_table" {
  value = aws_dynamodb_table.jobs.name
}
