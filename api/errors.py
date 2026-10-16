"""Public validation errors without submitted values or validator context."""

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException


def validation_response(request, exc: RequestValidationError):
    details = []
    fields = {}
    for error in exc.errors():
        kind = error["type"]
        message = {
            "missing": "This field is required.",
            "int_parsing": "Enter a whole number.",
            "int_from_float": "Enter a whole number.",
            "float_parsing": "Enter a number.",
            "greater_than": "Enter a value above the minimum.",
            "greater_than_equal": "Enter a value at or above the minimum.",
            "less_than_equal": "Enter a value at or below the maximum.",
            "string_pattern_mismatch": "Use the required format.",
            "string_too_short": "This value is too short.",
            "string_too_long": "This value is too long.",
            "literal_error": "Choose one of the allowed values.",
            "json_invalid": "Send a valid JSON request.",
        }.get(kind, "Check this value.")
        location = list(error["loc"])
        details.append({"loc": location, "msg": message, "type": kind})
        field = ".".join(str(part) for part in location if part not in ("body", "query", "path"))
        fields.setdefault(field or "request", message)
    return JSONResponse({"detail": details, "code": "validation_error", "fields": fields,
                         "request_id": getattr(request.state, "request_id", None)}, status_code=422)


def http_response(request, exc: HTTPException):
    code = {400: "invalid_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
            409: "conflict", 429: "rate_limited", 502: "upstream_unavailable",
            503: "service_unavailable", 504: "upstream_timeout"}.get(exc.status_code, "request_failed")
    return JSONResponse({"detail": exc.detail, "code": code,
                         "request_id": getattr(request.state, "request_id", None)},
                        status_code=exc.status_code, headers=exc.headers)
