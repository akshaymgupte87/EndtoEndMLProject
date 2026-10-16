variable "aws_region" {
  type    = string
  default = "us-west-2"
}

variable "project_name" {
  type    = string
  default = "recommender-demo"
}

variable "container_image" {
  type        = string
  description = "Pushed ECR image URI with an immutable build tag"
}

variable "model_bundle_key" {
  type        = string
  default     = "models/model_bundle.zip"
  description = "S3 key containing the exported model serving bundle"
}

variable "client_cidrs" {
  type        = list(string)
  description = "IPv4 CIDRs allowed to call the public demo API on port 8000; use your public IP /32"
  validation {
    condition     = length(var.client_cidrs) > 0 && alltrue([for cidr in var.client_cidrs : can(cidrnetmask(cidr)) && cidr != "0.0.0.0/0"])
    error_message = "Supply at least one valid IPv4 CIDR; use your own public IP /32 and do not allow 0.0.0.0/0."
  }
}

variable "bedrock_model_id" {
  type        = string
  default     = ""
  description = "Optional regional Bedrock foundation model ID for POST /explain/{user_idx}; empty disables explanations."
  validation {
    condition = var.bedrock_model_id == "" || (
      !startswith(var.bedrock_model_id, "arn:") &&
      !startswith(var.bedrock_model_id, "us.") &&
      !startswith(var.bedrock_model_id, "eu.") &&
      !startswith(var.bedrock_model_id, "global.")
    )
    error_message = "Supply a regional foundation model ID (for example amazon.nova-micro-v1:0), not a model ARN or cross-region inference-profile ID."
  }
}

variable "enable_sagemaker_mlflow" {
  type        = bool
  default     = false
  description = "Create a small SageMaker-managed MLflow tracking server with automatic Model Registry synchronization. This is billable while running."
}

variable "deploy_service" {
  type        = bool
  default     = true
  description = "Set false for the initial infrastructure apply before the image and model bundle are uploaded."
}
