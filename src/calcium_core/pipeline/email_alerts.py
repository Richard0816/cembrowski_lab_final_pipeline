"""SMTP / email-alert helpers for the calcium-imaging pipeline.

This module is deliberately kept isolated from :mod:`calcium_core.pipeline.workflow`
so that running the pipeline does not force anyone to pull in the SMTP machinery.
Import explicitly from here when you want alerts:

    from calcium_core.pipeline.email_alerts import (
        install_global_exception_handler,
        send_email,
        send_email_on_completion,
        send_email_on_failure,
    )
"""
from __future__ import annotations

import logging
import smtplib
import sys
import traceback
from datetime import datetime
from logging.handlers import SMTPHandler
from typing import Optional

# --- Email Configuration ----------------------------------------------------
# NOTE: use environment variables / a secrets manager for anything sensitive.
SMTP_SERVER = "smtp.gmail.com"       # e.g., smtp.gmail.com, SMTP.office365.com
SMTP_PORT = 587                      # typically 587 for TLS, 465 for SSL
SENDER_EMAIL = "richard.script.use@gmail.com"
RECIPIENT_EMAIL = "richardjiang2004@gmail.com"
EMAIL_PASSWORD = "uhau dvea emsk bair"  # app-specific password

# --- Setup Logging with SMTPHandler ----------------------------------------
error_logger = logging.getLogger(__name__)
error_logger.setLevel(logging.ERROR)

try:
    smtp_handler = SMTPHandler(
        mailhost=(SMTP_SERVER, SMTP_PORT),
        fromaddr=SENDER_EMAIL,
        toaddrs=[RECIPIENT_EMAIL],
        subject="CRITICAL Error in Python Script",
        credentials=(SENDER_EMAIL, EMAIL_PASSWORD),
        secure=(),  # Use secure=() for STARTTLS (port 587)
    )
    smtp_handler.setLevel(logging.ERROR)
    error_logger.addHandler(smtp_handler)
except smtplib.SMTPException as e:
    print(f"Failed to set up SMTP handler: {e}")


# --- Global Exception Handler ----------------------------------------------
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Log unhandled exceptions and send an email alert.

    Designed to be assigned to :data:`sys.excepthook`.  Delegates to the
    default excepthook after logging so that tracebacks still appear on the
    console.
    """
    error_logger.error(
        "An unhandled exception occurred:",
        exc_info=(exc_type, exc_value, exc_traceback),
    )
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def install_global_exception_handler() -> None:
    """Wire :func:`global_exception_handler` into :data:`sys.excepthook`."""
    sys.excepthook = global_exception_handler


# --- Plain-text email helpers ----------------------------------------------
def send_email(subject: str, body: str) -> None:
    """Send a plaintext email via SMTP (STARTTLS).

    Uses the module-level ``SMTP_*`` / ``SENDER_EMAIL`` / ``RECIPIENT_EMAIL`` /
    ``EMAIL_PASSWORD`` configuration.
    """
    msg = (
        f"From: {SENDER_EMAIL}\r\n"
        f"To: {RECIPIENT_EMAIL}\r\n"
        f"Subject: {subject}\r\n"
        f"\r\n"
        f"{body}\r\n"
    )

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, [RECIPIENT_EMAIL], msg.encode("utf-8", errors="replace"))


def send_email_on_completion(body: Optional[str] = None) -> None:
    """Send a standard 'pipeline finished' email.

    Swallows any exception and logs it so that a failed alert never masks a
    successful pipeline run.
    """
    if body is None:
        body = f"Pipeline finished at {datetime.now().isoformat()}"
    try:
        send_email(subject="Pipeline completed", body=body)
    except Exception as e:  # noqa: BLE001 - we never want alert code to crash
        print(f"[WARN] Final completion email failed: {e}")


def send_email_on_failure(context: str = "", exc: Optional[BaseException] = None) -> None:
    """Send a 'pipeline failed' email including traceback info when available."""
    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        body = f"{context}\n\n{tb}" if context else tb
    else:
        body = context or "Pipeline failed (no context provided)."
    try:
        send_email(subject="Pipeline FAILED", body=body)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Failure email failed: {e}")
