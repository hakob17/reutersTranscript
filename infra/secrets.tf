# Secret shells only — set the values out-of-band, never in Terraform state:
#   aws secretsmanager put-secret-value --secret-id <arn> --secret-string 'sk-ant-...'
#   aws secretsmanager put-secret-value --secret-id <arn> --secret-string 'hf_...'

resource "aws_secretsmanager_secret" "anthropic_api_key" {
  name        = "${var.project}/anthropic-api-key"
  description = "Claude API key for the attribution and chyron-reading stages"
}

resource "aws_secretsmanager_secret" "hf_token" {
  name        = "${var.project}/hf-token"
  description = "HuggingFace token (pyannote gated models); unused once weights are baked into the image"
}
