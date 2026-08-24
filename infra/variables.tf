variable "project" {
  description = "Name prefix for all resources"
  type        = string
  default     = "reuters-transcript"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "gpu_instance_types" {
  description = "Batch compute environment instance types (L4/A10G GPU)"
  type        = list(string)
  default     = ["g6.xlarge", "g5.xlarge"]
}

variable "use_spot" {
  description = "Run GPU workers on Spot (cuts GPU cost ~65%; jobs are short and idempotent)"
  type        = bool
  default     = true
}

variable "batch_max_vcpus" {
  description = "Ceiling for the GPU compute environment (4 vCPUs ~= one g6.xlarge)"
  type        = number
  default     = 16
}

variable "gpu_image_tag" {
  description = "Tag of the GPU worker image pushed to ECR"
  type        = string
  default     = "v1"
}

variable "review_timeout_hours" {
  description = "How long a needs_review video waits for an editor before the execution fails"
  type        = number
  default     = 72
}

variable "alert_email" {
  description = "Email for review notifications and operational alarms (empty = no subscription)"
  type        = string
  default     = ""
}
