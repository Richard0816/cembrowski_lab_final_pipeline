"""
End-of-run notification (SMTP email).

Safely optional: if SMTP settings are missing or the call fails we print a
warning and move on — long batch runs shouldn't abort because the network
flaked.

Defaults match the lab's notifier account. Override via environment variables
``CALCIUM_SMTP_SERVER``, ``CALCIUM_SMTP_PORT``, ``CALCIUM_SMTP_USER``,
``CALCIUM_SMTP_PASSWORD``, ``CALCIUM_SMTP_RECIPIENT``.
"""
from __future__ import annotations

import os
import smtplib
import ssl
from typing import Optional


DEFAULT_SMTP_SERVER = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 587
DEFAULT_SENDER = "richard.script.use@gmail.com"
DEFAULT_RECIPIENT = "richardjiang2004@gmail.com"


def _cfg(name: str, fallback: str) -> str:
    return os.environ.get(f"CALCIUM_{name}", fallback)


def send_email(
    subject: str,
    body: str,
    *,
    recipient: Optional[str] = None,
    password: Optional[str] = None,
    raise_on_error: bool = False,
) -> bool:
    """
    Send a plaintext email via STARTTLS SMTP. Returns ``True`` on success,
    ``False`` (and prints a warning) on failure.
    """
    server = _cfg("SMTP_SERVER", DEFAULT_SMTP_SERVER)
    port = int(_cfg("SMTP_PORT", str(DEFAULT_SMTP_PORT)))
    sender = _cfg("SMTP_USER", DEFAULT_SENDER)
    recipient = recipient or _cfg("SMTP_RECIPIENT", DEFAULT_RECIPIENT)
    password = password or _cfg("SMTP_PASSWORD", "")

    if not password:
        msg = "[notify] SMTP password not set (CALCIUM_SMTP_PASSWORD env var)"
        if raise_on_error:
            raise RuntimeError(msg)
        print(msg)
        return False

    email_body = (
        f"From: {sender}\r\n"
        f"To: {recipient}\r\n"
        f"Subject: {subject}\r\n"
        f"\r\n"
        f"{body}\r\n"
    )

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(server, port, timeout=30) as srv:
            srv.ehlo()
            srv.starttls(context=ctx)
            srv.ehlo()
            srv.login(sender, password)
            srv.sendmail(sender, [recipient],
                         email_body.encode("utf-8", errors="replace"))
        return True
    except Exception as ex:  # noqa: BLE001
        if raise_on_error:
            raise
        print(f"[notify] email send failed: {ex!r}")
        return False


__all__ = ["send_email"]
