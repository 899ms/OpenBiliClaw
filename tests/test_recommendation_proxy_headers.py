"""Contract tests for the main-API -> recommendation-process HTTP proxy.

Regression background
---------------------
``/api/recommendations/*`` is proxied to a dedicated recommendation process
(Unix socket on POSIX, loopback TCP on Windows) that runs the *same* auth
middleware. Its CSRF check compares the request ``Origin`` with the effective
``(scheme, host, port)``. If the proxy drops the original ``Host`` header, httpx
synthesises one from the backend URL (``localhost`` / ``127.0.0.1:<port>``), so a
same-origin browser POST authenticated by session cookie can never match its
``Origin`` and every write is rejected with ``403 {"error":"csrf"}``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from openbiliclaw.api.app import create_app
from openbiliclaw.api.auth import AuthGate
from openbiliclaw.config import ApiAuthConfig, Config
from openbiliclaw.recommendation_runtime import (
    RECOMMENDATION_PORT_ENV,
    RECOMMENDATION_SOCK_ENV,
)

if TYPE_CHECKING:
    from pathlib import Path

_ORIGIN = "http://127.0.0.1:8420"
_APPEND_PATH = "/api/recommendations/append"
_TCP_PORT = "18424"
_SOCK_PATH = "/tmp/openbiliclaw-test/recommendation.sock"


class _FakeAsyncHTTPTransport:
    """Records the transport kwargs the proxy builds (uds vs loopback TCP)."""

    last_kwargs: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs

    @classmethod
    def reset(cls) -> None:
        cls.last_kwargs = {}


class _FakeAsyncClient:
    """Captures exactly what the proxy would put on the wire.

    ``Host`` is filled in from the request URL when absent, mirroring httpx, so
    a regression that strips the header shows up as ``host: localhost``.
    """

    forwards: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        # The proxy builds the client with transport/timeout kwargs; the
        # transport kwargs are asserted via _FakeAsyncHTTPTransport instead.
        self.kwargs = kwargs

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
    ) -> httpx.Response:
        sent = {key.lower(): value for key, value in (headers or {}).items()}
        sent.setdefault("host", urlsplit(url).netloc)
        type(self).forwards.append(
            {"method": method, "url": url, "headers": sent, "content": content}
        )
        return httpx.Response(200, json={"items": [], "pool_status": None})


@pytest.fixture(autouse=True)
def _reset_capture() -> None:
    _FakeAsyncClient.forwards = []
    _FakeAsyncHTTPTransport.reset()


def _build_proxied_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """Build the main API app with the recommendation proxy enabled."""
    config = Config(data_dir=str(tmp_path))
    monkeypatch.setattr("openbiliclaw.config.load_config", lambda *_a, **_kw: config)
    monkeypatch.delenv("OPENBILICLAW_RECOMMENDATION_ONLY", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", _FakeAsyncHTTPTransport)
    return create_app(
        memory_manager=SimpleNamespace(load_discovery_runtime_state=lambda: {}),
        database=SimpleNamespace(),
        soul_engine=SimpleNamespace(),
        runtime_controller=SimpleNamespace(event_hub=None),
        recommendation_engine=SimpleNamespace(),
    )


def _forwarded_request(headers: dict[str, str]) -> Request:
    """Rebuild the request the recommendation process receives from the proxy."""
    scope: dict[str, Any] = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": _APPEND_PATH,
        "raw_path": _APPEND_PATH.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(key.encode(), value.encode()) for key, value in headers.items()],
        "client": None,  # Unix-socket peers have no client address
        "server": ("localhost", 80),
    }
    return Request(scope)


def _post_append(client: TestClient) -> Any:
    return client.post(
        _APPEND_PATH,
        json={"excluded_bvids": []},
        headers={
            "X-OBC-Auth": "1",
            "Origin": _ORIGIN,
            "Cookie": "obc_session=test-token",
        },
    )


def test_proxy_forwards_original_host_and_origin(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(RECOMMENDATION_SOCK_ENV, _SOCK_PATH)
    monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)
    app = _build_proxied_app(monkeypatch, tmp_path)

    response = _post_append(TestClient(app, base_url=_ORIGIN))

    assert response.status_code == 200
    assert len(_FakeAsyncClient.forwards) == 1
    forwarded = _FakeAsyncClient.forwards[0]
    # The recommendation process must see the browser's own authority, not the
    # backend URL httpx would otherwise invent.
    assert forwarded["headers"]["host"] == "127.0.0.1:8420"
    assert forwarded["headers"]["origin"] == _ORIGIN


def test_proxy_hops_do_not_forward_length_or_connection(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(RECOMMENDATION_SOCK_ENV, _SOCK_PATH)
    monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)
    app = _build_proxied_app(monkeypatch, tmp_path)

    _post_append(TestClient(app, base_url=_ORIGIN))

    headers = _FakeAsyncClient.forwards[0]["headers"]
    assert "content-length" not in headers
    assert "connection" not in headers


def test_proxy_uses_unix_socket_transport_on_posix(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(RECOMMENDATION_SOCK_ENV, _SOCK_PATH)
    monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)
    app = _build_proxied_app(monkeypatch, tmp_path)

    _post_append(TestClient(app, base_url=_ORIGIN))

    assert _FakeAsyncClient.forwards[0]["url"] == f"http://localhost{_APPEND_PATH}"
    # The socket path (not just the URL shape) is what routes the hop to the
    # recommendation process on POSIX.
    assert _FakeAsyncHTTPTransport.last_kwargs == {"uds": _SOCK_PATH}


def test_proxy_uses_loopback_tcp_transport_when_port_is_configured(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(RECOMMENDATION_SOCK_ENV, raising=False)
    monkeypatch.setenv(RECOMMENDATION_PORT_ENV, _TCP_PORT)
    app = _build_proxied_app(monkeypatch, tmp_path)

    _post_append(TestClient(app, base_url=_ORIGIN))

    forwarded = _FakeAsyncClient.forwards[0]
    assert forwarded["url"] == f"http://127.0.0.1:{_TCP_PORT}{_APPEND_PATH}"
    assert forwarded["headers"]["host"] == "127.0.0.1:8420"
    # Loopback TCP carries no Unix socket.
    assert _FakeAsyncHTTPTransport.last_kwargs == {}


@pytest.mark.parametrize("transport", ["unix", "tcp"])
def test_proxied_request_passes_recommendation_process_csrf(
    monkeypatch, tmp_path: Path, transport: str
) -> None:
    """The whole point of forwarding Host: the rec process accepts the request."""
    if transport == "unix":
        monkeypatch.setenv(RECOMMENDATION_SOCK_ENV, _SOCK_PATH)
        monkeypatch.delenv(RECOMMENDATION_PORT_ENV, raising=False)
    else:
        monkeypatch.delenv(RECOMMENDATION_SOCK_ENV, raising=False)
        monkeypatch.setenv(RECOMMENDATION_PORT_ENV, _TCP_PORT)
    app = _build_proxied_app(monkeypatch, tmp_path)

    _post_append(TestClient(app, base_url=_ORIGIN))

    gate = AuthGate(ApiAuthConfig(enabled=True, session_secret="test-secret"), None)
    forwarded = _FakeAsyncClient.forwards[0]["headers"]
    assert gate.csrf_ok(_forwarded_request(forwarded)) is True


def test_recommendation_process_csrf_rejects_synthesised_backend_host(
    monkeypatch, tmp_path: Path
) -> None:
    """Guard: dropping Host reproduces the reported 403 (the pre-fix behaviour)."""
    gate = AuthGate(ApiAuthConfig(enabled=True, session_secret="test-secret"), None)

    broken = {
        "host": "localhost",  # what httpx invents once Host is stripped
        "origin": _ORIGIN,
        "x-obc-auth": "1",
        "cookie": "obc_session=test-token",
    }

    assert gate.csrf_ok(_forwarded_request(broken)) is False
