resource "aws_lb" "main" {
  name                       = local.name
  load_balancer_type         = "application"
  internal                   = false
  security_groups            = [aws_security_group.alb.id]
  subnets                    = aws_subnet.public[*].id
  drop_invalid_header_fields = true
  idle_timeout               = 60
  enable_deletion_protection = false # on-demand stack: destroyed after a few hours
  # checkov:skip=CKV_AWS_91:access logs need an extra bucket; CloudWatch metrics and app logs suffice here
  # checkov:skip=CKV_AWS_150:deletion protection would block the auto-destroy of this on-demand stack
  # checkov:skip=CKV2_AWS_20:HTTP only; HTTPS needs a domain and an ACM certificate (see README)
  # checkov:skip=CKV2_AWS_28:WAF costs money; demo API without sensitive data
}

resource "aws_lb_target_group" "api" {
  name                 = "${local.name}-api"
  port                 = local.container_port
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.main.id
  deregistration_delay = 20 # matches uvicorn --timeout-graceful-shutdown 20
  # checkov:skip=CKV_AWS_378:ALB-to-task hop stays inside the VPC; TLS would need certificates in the container

  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 15
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"
  # checkov:skip=CKV_AWS_2:HTTPS needs a domain and an ACM certificate; documented in the README
  # checkov:skip=CKV_AWS_103:no TLS listener (see above)
  # checkov:skip=CKV2_AWS_20:no TLS listener to redirect to (see above)

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}
