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
