output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
}

output "training_ecr_repository_url" {
  value = aws_ecr_repository.training.repository_url
}

output "sagemaker_pipeline_role_arn" {
  value = aws_iam_role.sagemaker_pipeline.arn
}

output "sagemaker_training_role_arn" {
  value = aws_iam_role.sagemaker_training.arn
}

output "model_bucket" {
  value = aws_s3_bucket.models.bucket
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  value = aws_ecs_service.api.name
}

output "default_vpc_id" {
  value = data.aws_vpc.default.id
}

output "api_security_group_id" {
  value = aws_security_group.api.id
}

output "api_port" {
  value = 8000
}

output "model_upload_command" {
  value = "aws s3 cp <model_bundle.zip> s3://${aws_s3_bucket.models.bucket}/${var.model_bundle_key}"
}

output "cloudwatch_dashboard_name" {
  value = aws_cloudwatch_dashboard.api.dashboard_name
}

output "sagemaker_mlflow_arn" {
  value       = var.enable_sagemaker_mlflow ? aws_sagemaker_mlflow_tracking_server.main[0].arn : null
  description = "ARN for the optional SageMaker-managed MLflow tracking server."
}

output "sagemaker_mlflow_url" {
  value       = var.enable_sagemaker_mlflow ? aws_sagemaker_mlflow_tracking_server.main[0].tracking_server_url : null
  description = "MLflow UI URL for the optional SageMaker-managed tracking server."
}
