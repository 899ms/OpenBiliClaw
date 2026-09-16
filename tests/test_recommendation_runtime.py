"""Tests for the cross-platform recommendation process transport."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from openbiliclaw import recommendation_runtime
from openbiliclaw.recommendation_runtime import (
    DEFAULT_RECOMMENDATION_PORT,
    RECOMMENDATION_PORT_ENV,
    RECOMMENDATION_SOCK_ENV,
    ensure_recommendation_transport_env,
    recommendation_sock_from_data_path,
    recommendation_transport_enabled,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_recommendation_sock_from_data_path(tmp_path: Path) -> None:
    assert recommendation_sock_from_data_path(tmp_path) == str(
        tmp_path / "runtime" / "recommendation.sock"
    )


def test_ensure_recommendation_transport_env_posix(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        RECOMMENDATION_SOCK_ENV, recommendation_sock_from_data_path(tmp_path)
    )
    monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)
    monkeypatch.setattr(recommendation_runtime.os, "name", "posix")

    description = ensure_recommendation_transport_env(tmp_path)

    assert description == f"Unix socket {recommendation_sock_from_data_path(tmp_path)}"
    assert RECOMMENDATION_SOCK_ENV in recommendation_runtime.os.environ
    assert RECOMMENDATION_PORT_ENV not in recommendation_runtime.os.environ
    assert recommendation_transport_enabled() is True


def test_ensure_recommendation_transport_env_windows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(RECOMMENDATION_SOCK_ENV, raising=False)
    monkeypatch.setenv(RECOMMENDATION_PORT_ENV, str(DEFAULT_RECOMMENDATION_PORT))
    monkeypatch.setattr(recommendation_runtime.os, "name", "nt")

    description = ensure_recommendation_transport_env(tmp_path)

    assert description == f"TCP 127.0.0.1:{DEFAULT_RECOMMENDATION_PORT}"
    assert RECOMMENDATION_PORT_ENV in recommendation_runtime.os.environ
    assert RECOMMENDATION_SOCK_ENV not in recommendation_runtime.os.environ
    assert recommendation_transport_enabled() is True


def test_recommendation_transport_enabled_false_without_env(monkeypatch) -> None:
    monkeypatch.delenv(RECOMMENDATION_SOCK_ENV, raising=False)
    monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)

    assert recommendation_transport_enabled() is False


def test_ensure_recommendation_transport_env_posix_with_explicit_port(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(RECOMMENDATION_SOCK_ENV, raising=False)
    monkeypatch.setenv(RECOMMENDATION_PORT_ENV, "18424")
    monkeypatch.setattr(recommendation_runtime.os, "name", "posix")

    description = ensure_recommendation_transport_env(tmp_path)

    assert description == "TCP 127.0.0.1:18424"
    assert RECOMMENDATION_PORT_ENV in recommendation_runtime.os.environ
    assert RECOMMENDATION_SOCK_ENV not in recommendation_runtime.os.environ


def test_recommendation_server_windows_binds_tcp(monkeypatch) -> None:
    import openbiliclaw.recommendation_server as recommendation_server

    calls: list[dict[str, object]] = []
    fake_app = object()
    # Transport tests must not depend on the developer machine's on-disk LLM
    # config; the readiness gate itself is covered by test_worker_degraded_boot.
    monkeypatch.setattr(recommendation_server, "wait_for_buildable_llm", lambda: None)
    monkeypatch.setattr(recommendation_server.os, "name", "nt")
    monkeypatch.setattr(recommendation_server, "create_app", lambda: fake_app)
    monkeypatch.setattr(
        recommendation_server.uvicorn,
        "run",
        lambda *args, **kwargs: calls.append({"app": args[0], **kwargs}),
    )
    monkeypatch.setenv(RECOMMENDATION_PORT_ENV, str(DEFAULT_RECOMMENDATION_PORT))

    recommendation_server.main()

    assert calls == [
        {
            "app": fake_app,
            "host": "127.0.0.1",
            "port": DEFAULT_RECOMMENDATION_PORT,
            "log_level": "info",
        }
    ]


@pytest.mark.skipif(recommendation_runtime.os.name == "nt", reason="POSIX Unix socket path")
def test_recommendation_server_posix_binds_unix_socket(
    monkeypatch, tmp_path: Path
) -> None:
    import openbiliclaw.recommendation_server as recommendation_server

    calls: list[dict[str, object]] = []
    fake_app = object()
    sock = recommendation_sock_from_data_path(tmp_path)
    # Same as the Windows transport test: binding behavior is independent of
    # the on-disk LLM config readiness gate.
    monkeypatch.setattr(recommendation_server, "wait_for_buildable_llm", lambda: None)
    monkeypatch.setattr(recommendation_server, "create_app", lambda: fake_app)
    monkeypatch.setattr(
        recommendation_server.uvicorn,
        "run",
        lambda *args, **kwargs: calls.append({"app": args[0], **kwargs}),
    )
    monkeypatch.setenv(RECOMMENDATION_SOCK_ENV, sock)
    monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)

    recommendation_server.main()

    assert calls == [{"app": fake_app, "uds": sock, "log_level": "info"}]
