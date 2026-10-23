import json
import hmac
import os
import traceback
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from backend.model_provider import required_model_settings


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
HOST = os.getenv("HOST", "0.0.0.0").strip() or "0.0.0.0"
PORT = int(os.getenv("PORT", "8100"))
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", str(1024 * 1024)))
MAX_MESSAGE_CHARACTERS = int(os.getenv("MAX_MESSAGE_CHARACTERS", "20000"))
MAX_HISTORY_MESSAGES = 100
MAX_HISTORY_CHARACTERS = 60_000
SERVICE_KEY_HEADER = "X-AI-Service-Key"
REQUIRED_READY_SETTINGS = (
    "MILVUS_URI",
    "MILVUS_COLLECTION",
    "AI_SERVICE_SECRET_KEY",
)
CORS_ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
}


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _log(message: str) -> None:
    print(f"[{_utc_now()}] {message}", flush=True)


def _get_assistant() -> Any:
    package_path = str(BASE_DIR)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    from backend.agents.orchestrator.orchestrator_agent import ask_scor_agent

    return ask_scor_agent


def _normalize_history(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("history must be an array.")
    if len(value) > MAX_HISTORY_MESSAGES:
        raise ValueError(f"history cannot exceed {MAX_HISTORY_MESSAGES} messages.")

    normalized: list[dict[str, str]] = []
    total_characters = 0
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"history[{index}] must be an object.")
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"}:
            raise ValueError(f"history[{index}].role must be 'user' or 'assistant'.")
        if not content:
            raise ValueError(f"history[{index}].content cannot be empty.")
        total_characters += len(content)
        if total_characters > MAX_HISTORY_CHARACTERS:
            raise ValueError(
                f"history cannot exceed {MAX_HISTORY_CHARACTERS} total characters."
            )
        normalized.append({"role": role, "content": content})
    return normalized


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "ScorAssistantBackend/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/health", "/api/health"}:
            self._send_json({"status": "ok", "service": "scor-assistant-backend"})
            return
        if parsed.path in {"/ready", "/api/ready"}:
            missing = [name for name in (*required_model_settings(), *REQUIRED_READY_SETTINGS) if not os.getenv(name, "").strip()]
            status = HTTPStatus.OK if not missing else HTTPStatus.SERVICE_UNAVAILABLE
            self._send_json(
                {
                    "status": "ready" if not missing else "not_ready",
                    "service": "scor-assistant-backend",
                    "missing_settings": missing,
                },
                status=status,
            )
            return
        if parsed.path == "/":
            self._send_json(
                {
                    "service": "scor-assistant-backend",
                    "status": "ok",
                    "storage": "stateless",
                    "endpoints": {
                        "chat": "/api/chat",
                        "assessment_reports": "/api/assessment-reports/generate",
                        "health": "/health",
                        "readiness": "/ready",
                    },
                }
            )
            return
        self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        protected_paths = {
            "/api/chat",
            "/api/messages",
            "/api/assessment-reports/generate",
        }
        if parsed.path not in protected_paths:
            self._send_json({"error": "Not found."}, status=HTTPStatus.NOT_FOUND)
            return
        if not self._authorize_service_request():
            return

        if parsed.path == "/api/assessment-reports/generate":
            body, error, error_status = self._read_json_body()
            if error:
                self._send_json({"error": error}, status=error_status)
                return

            from backend.assessment_ai_reports.service import (
                AssessmentReportConfigurationError,
                AssessmentReportGenerationError,
                AssessmentReportValidationError,
                generate_assessment_ai_report_response,
            )

            try:
                result = generate_assessment_ai_report_response(body)
            except AssessmentReportValidationError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except AssessmentReportConfigurationError:
                _log("Assessment report service is not configured.")
                self._send_json(
                    {"error": "Assessment report service is not configured."},
                    status=HTTPStatus.SERVICE_UNAVAILABLE,
                )
                return
            except AssessmentReportGenerationError as exc:
                cause = exc.__cause__
                if cause is not None:
                    _log(
                        "Assessment report generation failed: "
                        f"{type(cause).__name__}: {cause}"
                    )
                else:
                    _log(f"Assessment report generation failed: {exc}")
                self._send_json(
                    {"error": "Assessment report generation failed."},
                    status=HTTPStatus.BAD_GATEWAY,
                )
                return

            self._send_json(result)
            return
        body, error, error_status = self._read_json_body()
        if error:
            self._send_json({"error": error}, status=error_status)
            return

        message = str(body.get("message", "")).strip()
        if not message:
            self._send_json({"error": "message is required."}, status=HTTPStatus.BAD_REQUEST)
            return
        if len(message) > MAX_MESSAGE_CHARACTERS:
            self._send_json(
                {"error": f"message cannot exceed {MAX_MESSAGE_CHARACTERS} characters."},
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        try:
            history = _normalize_history(body.get("history"))
            fallback_history = _normalize_history(body.get("fallback_history"))
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        previous_response_id_value = body.get("previous_response_id")
        previous_response_id = (
            str(previous_response_id_value).strip()
            if previous_response_id_value is not None
            else None
        )
        if previous_response_id == "":
            previous_response_id = None
        if history and previous_response_id:
            self._send_json(
                {"error": "Provide either history or previous_response_id, not both."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        project_arguments: dict[str, Any] = {}
        if "project_id" in body:
            raw_project_id = body["project_id"]
            if isinstance(raw_project_id, bool) or not (isinstance(raw_project_id, int) or isinstance(raw_project_id, str) and raw_project_id.strip().isdigit()):
                self._send_json({"error": "project_id must be a positive integer."}, status=HTTPStatus.BAD_REQUEST)
                return
            project_id = int(raw_project_id)
            if project_id <= 0:
                self._send_json({"error": "project_id must be a positive integer."}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                from backend.company_context_repository import get_project_folder_id, get_company_context
                folder_id = get_project_folder_id(project_id)
                if folder_id is None:
                    self._send_json({"error": "Project not found."}, status=HTTPStatus.NOT_FOUND)
                    return
                project_arguments = {
                    "project_id": project_id,
                    "project_folder_id": folder_id,
                    "company_context": get_company_context(folder_id),
                }
            except Exception as exc:
                _log(f"Project context lookup failed: {type(exc).__name__}")
                self._send_json({"error": "Project data is unavailable."}, status=HTTPStatus.SERVICE_UNAVAILABLE)
                return

        ask_scor_agent = _get_assistant()
        try:
            result = ask_scor_agent(
                user_message=message,
                previous_response_id=previous_response_id,
                conversation_history=history,
                fallback_history=fallback_history,
                **project_arguments,
            )
        except Exception as exc:
            _log(f"Assistant request failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._send_json(
                {"error": "Assistant request failed. Check the application logs."},
                status=HTTPStatus.BAD_GATEWAY,
            )
            return

        self._send_json(
            {
                "answer": result["answer"],
                "response_id": result.get("response_id"),
                "continuation_mode": result.get(
                    "continuation_mode",
                    "response_id" if result.get("response_id") else "history",
                ),
            }
        )

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        _log(f"HTTP {self.command} {self.path} - {format % args}")

    def _authorize_service_request(self) -> bool:
        expected_key = os.getenv("AI_SERVICE_SECRET_KEY", "").strip()
        if not expected_key:
            _log("Service authentication is not configured.")
            self._send_json(
                {"error": "Service authentication is not configured."},
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return False

        provided_key = self.headers.get(SERVICE_KEY_HEADER, "").strip()
        if not provided_key or not hmac.compare_digest(provided_key, expected_key):
            self._send_json(
                {"error": "Unauthorized."},
                status=HTTPStatus.UNAUTHORIZED,
            )
            return False
        return True

    def _read_json_body(
        self,
    ) -> tuple[dict[str, Any], str | None, HTTPStatus]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return {}, "Invalid Content-Length header.", HTTPStatus.BAD_REQUEST
        if content_length <= 0:
            return {}, "A JSON request body is required.", HTTPStatus.BAD_REQUEST
        if content_length > MAX_REQUEST_BYTES:
            return (
                {},
                f"Request body cannot exceed {MAX_REQUEST_BYTES} bytes.",
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
        try:
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}, "Invalid JSON request body.", HTTPStatus.BAD_REQUEST
        if not isinstance(body, dict):
            return {}, "JSON request body must be an object.", HTTPStatus.BAD_REQUEST
        return body, None, HTTPStatus.OK

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        response = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(response)

    def _send_cors_headers(self) -> None:
        request_origin = self.headers.get("Origin", "").strip()
        if "*" in CORS_ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", "*")
        elif request_origin and request_origin in CORS_ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", request_origin)
            self.send_header("Vary", "Origin")
        if CORS_ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header(
                "Access-Control-Allow-Headers",
                f"Content-Type, Authorization, {SERVICE_KEY_HEADER}",
            )


def run_server(host: str = HOST, port: int = PORT) -> None:
    server = ThreadingHTTPServer((host, port), ChatHandler)
    _log(f"Stateless SCOR assistant backend running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
