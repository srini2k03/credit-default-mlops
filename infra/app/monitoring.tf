resource "aws_sns_topic" "alerts" {
  name = "${local.name}-alerts"
  # checkov:skip=CKV_AWS_26:CloudWatch alarms cannot publish to a topic encrypted with the AWS-managed key, and a CMK costs $1/month
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

locals {
  alb_dims = { LoadBalancer = aws_lb.main.arn_suffix }
  tg_dims = {
    LoadBalancer = aws_lb.main.arn_suffix
    TargetGroup  = aws_lb_target_group.api.arn_suffix
  }
  alarm_actions = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "http_5xx_count" {
  alarm_name          = "${local.name}-5xx-count"
  alarm_description   = "ALB and target 5XX responses: 5 or more in 5 minutes"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  threshold           = 5
  evaluation_periods  = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions

  metric_query {
    id          = "errors"
    expression  = "FILL(elb, 0) + FILL(target, 0)"
    label       = "5XX responses"
    return_data = true
  }
  metric_query {
    id = "elb"
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "HTTPCode_ELB_5XX_Count"
      dimensions  = local.alb_dims
      period      = 300
      stat        = "Sum"
    }
  }
  metric_query {
    id = "target"
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "HTTPCode_Target_5XX_Count"
      dimensions  = local.alb_dims
      period      = 300
      stat        = "Sum"
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "http_5xx_rate" {
  alarm_name          = "${local.name}-5xx-rate"
  alarm_description   = "5XX responses above ${var.error_rate_threshold_percent}% of requests over 5 minutes"
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.error_rate_threshold_percent
  evaluation_periods  = 1
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions

  metric_query {
    id          = "rate"
    expression  = "IF(requests > 0, 100 * (FILL(elb, 0) + FILL(target, 0)) / requests, 0)"
    label       = "5XX rate (%)"
    return_data = true
  }
  metric_query {
    id = "requests"
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "RequestCount"
      dimensions  = local.alb_dims
      period      = 300
      stat        = "Sum"
    }
  }
  metric_query {
    id = "elb"
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "HTTPCode_ELB_5XX_Count"
      dimensions  = local.alb_dims
      period      = 300
      stat        = "Sum"
    }
  }
  metric_query {
    id = "target"
    metric {
      namespace   = "AWS/ApplicationELB"
      metric_name = "HTTPCode_Target_5XX_Count"
      dimensions  = local.alb_dims
      period      = 300
      stat        = "Sum"
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "latency_p95" {
  alarm_name          = "${local.name}-latency-p95"
  alarm_description   = "Target response time p95 above ${var.latency_p95_threshold_seconds * 1000} ms in 3 of 5 minutes"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "TargetResponseTime"
  dimensions          = local.tg_dims
  extended_statistic  = "p95"
  period              = 60
  evaluation_periods  = 5
  datapoints_to_alarm = 3
  comparison_operator = "GreaterThanThreshold"
  threshold           = var.latency_p95_threshold_seconds
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

resource "aws_cloudwatch_metric_alarm" "unhealthy_hosts" {
  alarm_name          = "${local.name}-unhealthy-hosts"
  alarm_description   = "At least one unhealthy target for 3 minutes"
  namespace           = "AWS/ApplicationELB"
  metric_name         = "UnHealthyHostCount"
  dimensions          = local.tg_dims
  statistic           = "Maximum"
  period              = 60
  evaluation_periods  = 3
  comparison_operator = "GreaterThanThreshold"
  threshold           = 0
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  ok_actions          = local.alarm_actions
}

# ---------------------------------------------------------------- stack record for the workflows

# Written early in the apply (depends only on time_static), so even a half-created stack has a
# creation time for the auto-destroy job.
resource "aws_ssm_parameter" "created_at" {
  name        = "/${var.project}/app/created_at"
  description = "Creation time of the app stack (RFC 3339), read by the auto-destroy job"
  type        = "String"
  value       = time_static.created.rfc3339
}

resource "aws_ssm_parameter" "stack" {
  name        = local.ssm_stack_parameter
  description = "App stack facts for deploy-aws.yml and the auto-destroy job"
  type        = "String" # standard tier: free
  value = jsonencode({
    created_at     = time_static.created.rfc3339
    alb_url        = "http://${aws_lb.main.dns_name}"
    cluster        = aws_ecs_cluster.main.name
    service        = aws_ecs_service.api.name
    task_family    = aws_ecs_task_definition.api.family
    container_name = local.container_name
    log_group      = aws_cloudwatch_log_group.api.name
    model_s3_uri   = local.model_s3_uri
  })
  # checkov:skip=CKV2_AWS_34:not a secret (public URL and resource names)
}
