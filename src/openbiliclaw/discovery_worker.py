"""Background discovery/runtime worker.

Builds the same runtime components as the main API, but instead of serving
HTTP it only runs ``ContinuousRefreshController.run_forever()``. This keeps
discovery / eval / producers / cover prefetch away from the main API event
loop while still letting the API stay fast for recommendations and chat.

When the on-disk config cannot build any LLM instance yet (fresh install, or
a broken key), this worker no longer dies with an uncaught ``RuntimeError``
that popped a PyInstaller error dialog in windowed builds: it probes
``config.toml`` cheaply and retries until the user saves a usable key, then
starts the normal runtime loop without an app restart.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from openbiliclaw.api.app import create_app

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Retry cadence while the config cannot build an LLM registry (construction
# only, no network) or the runtime context comes up degraded.
DEFAULT_LLM_PROBE_RETRY_SECONDS = 15.0


def _probe_llm_registry() -> None:
    """Build the LLM registry from the current on-disk config."""
    from openbiliclaw.config import load_config
    from openbiliclaw.llm.registry import build_llm_registry

    build_llm_registry(load_config())


def _release_failed_context(ctx: Any) -> None:
    """Close the database of a context built for a failed attempt, best effort."""
    database = getattr(ctx, "database", None)
    close = getattr(database, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.exception("Failed to close discovery worker context")


def run_discovery_worker(
    *,
    retry_interval_seconds: float = DEFAULT_LLM_PROBE_RETRY_SECONDS,
    probe: Callable[[], None] | None = None,
    app_factory: Callable[[], Any] = create_app,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run the discovery runtime loop, waiting out an unbuildable LLM config.

    ``probe``/``app_factory``/``sleep`` are injectable for tests; production
    calls use the real config probe, ``create_app`` and ``time.sleep``.
    """
    probe_fn = probe or _probe_llm_registry
    retry_interval = max(1.0, float(retry_interval_seconds))
    waiting_logged = False
    while True:
        try:
            probe_fn()
        except Exception as exc:
            if not waiting_logged:
                logger.warning(
                    "Discovery worker is waiting for a buildable LLM configuration (%s); "
                    "retrying every %.0fs",
                    exc,
                    retry_interval,
                )
                waiting_logged = True
            sleep(retry_interval)
            continue

        try:
            app = app_factory()
            ctx = getattr(app.state, "runtime_context", None)
            controller = getattr(ctx, "runtime_controller", None)
            run_forever = getattr(controller, "run_forever", None)
        except Exception:
            logger.exception(
                "Discovery worker startup failed; retrying in %.0fs",
                retry_interval,
            )
            sleep(retry_interval)
            continue

        if run_forever is None:
            logger.error(
                "Discovery worker got a runtime context without "
                "runtime_controller.run_forever (degraded=%s, reason=%s); retrying in %.0fs",
                bool(getattr(ctx, "degraded", False)),
                str(getattr(ctx, "degraded_reason", "") or "unknown"),
                retry_interval,
            )
            _release_failed_context(ctx)
            sleep(retry_interval)
            continue

        if waiting_logged:
            logger.info("Discovery worker LLM configuration recovered; starting runtime loop")
        asyncio.run(run_forever())
        return


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("OPENBILICLAW_FULL_WORKER", "1")
    run_discovery_worker()


if __name__ == "__main__":
    main()
