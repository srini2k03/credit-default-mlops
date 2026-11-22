output "alb_url" {
  value = "http://${aws_lb.main.dns_name}"
}

output "cluster" {
  value = aws_ecs_cluster.main.name
}

output "service" {
  value = aws_ecs_service.api.name
}

output "log_group" {
  value = aws_cloudwatch_log_group.api.name
}

output "created_at" {
  value = time_static.created.rfc3339
}

output "model_s3_uri" {
  value = local.model_s3_uri
}

output "alerts_topic_arn" {
  value = aws_sns_topic.alerts.arn
}
