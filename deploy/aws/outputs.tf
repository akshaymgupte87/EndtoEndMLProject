output "ecr_repository_url" {
  value = aws_ecr_repository.api.repository_url
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

output "cloudwatch_dashboard_name" {
  value = aws_cloudwatch_dashboard.api.dashboard_name
}
