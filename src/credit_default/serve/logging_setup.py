"""Service logging: plain text (default) or one JSON object per line on stdout.

Environment:
  LOG_FORMAT  text (default) | json. JSON lines are what CloudWatch Logs (awslogs driver) ingests
              and queries field by field; text keeps local runs and Render as they were.
  LOG_LEVEL   default INFO

``RequestContextMiddleware`` gives every HTTP request an id (the caller's ``X-Request-ID`` when it
is a safe token, else a new one), echoes it in the response header, keeps it in a context
variable so every log line written while serving the request carries it, and logs one line per
request with method, path, status and latency.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import UTC, datetime

from starlette.datastructures import Headers, MutableHeaders

REQUEST_ID_HEADER = "X-Request-ID"
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

# Accept a caller's id only if it cannot break a log line or a header.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")
# Attributes every LogRecord has; anything else was passed via ``extra=`` and becomes a field.
# uvicorn adds ``color_message`` (the message with ANSI colours), which is noise in JSON.
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {
    "message",
    "asctime",
    "taskName",
    "color_message",
}
_HANDLER_FLAG = "_credit_default_json"

log = logging.getLogger("credit_default.serve.access")


class RequestIdFilter(logging.Filter):
    """Copy the current request id (if any) onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "request_id", None) is None:
            record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_") and value is not None:
                entry[key] = value
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            entry["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(entry, default=str)


def json_handler(stream=None) -> logging.Handler:
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())
    setattr(handler, _HANDLER_FLAG, True)
    return handler


def configure_logging(fmt: str | None = None, level: str | None = None) -> str:
    """Set up the root logger. Text mode is exactly the previous ``basicConfig`` call."""
    fmt = (fmt or os.environ.get("LOG_FORMAT", "text")).lower()
    level = level or os.environ.get("LOG_LEVEL", "INFO")
    if fmt != "json":
        logging.basicConfig(level=level)
        return "text"

    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, _HANDLER_FLAG, False):
            root.removeHandler(h)
    root.addHandler(json_handler())
    root.setLevel(level)
    # uvicorn installs its own text handlers before importing the app: route its server and
    # error logs through the JSON handler, and silence its access log, which the middleware's
    # per-request line replaces.
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
    access.disabled = True
    return "json"


class RequestContextMiddleware:
    """Pure ASGI middleware: request id in, request id out, one log line per request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        rid = incoming if incoming and _SAFE_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(rid)
        start = time.perf_counter()
        status = 500

        async def send_with_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = rid
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            log.exception("unhandled error on %s %s", scope.get("method"), scope.get("path"))
            raise
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            log.info(
                "%s %s %s %.2fms request_id=%s",
                scope.get("method"),
                scope.get("path"),
                status,
                latency_ms,
                rid,
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status,
                    "latency_ms": latency_ms,
                    "request_id": rid,
                },
            )
            request_id_var.reset(token)
