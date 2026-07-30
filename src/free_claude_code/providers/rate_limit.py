"""Provider-owned upstream rate limiting and retry policy."""

import asyncio
import random
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

import httpx
import openai
from loguru import logger

from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.rate_limit import StrictSlidingWindowLimiter
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.failure_policy import (
    ProviderFailureOverride,
    cooldown_seconds_from_exception,
    retryable_upstream_status,
    retryable_upstream_transport_error,
)

KeyFailureCallback = Callable[[FailureKind, float], None]

T = TypeVar("T")

UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS = 5
DEFAULT_UPSTREAM_MAX_RETRIES = UPSTREAM_TRANSIENT_TOTAL_ATTEMPTS - 1


class ProviderRateLimiter:
    """
    Rate limiter owned by one provider instance.

    Blocks that provider's requests when a rate-limit error is encountered
    (reactive) and throttles its requests with a strict rolling window
    (proactive).

    Optionally enforces a max_concurrency cap: at most N provider streams
    may be open simultaneously, independent of the sliding window.

    Proactive limits - throttles requests to stay within API limits.
    Reactive limits - pauses all requests when a 429 or 5xx retry backoff is active.
    Concurrency limit - caps simultaneously open streams.
    """

    def __init__(
        self,
        rate_limit: int = 40,
        rate_window: float = 60.0,
        max_concurrency: int = 5,
    ):
        if rate_limit <= 0:
            raise ValueError("rate_limit must be > 0")
        if rate_window <= 0:
            raise ValueError("rate_window must be > 0")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")

        self._rate_limit = rate_limit
        self._rate_window = float(rate_window)
        self._max_concurrency = max_concurrency
        self._proactive_limiter = StrictSlidingWindowLimiter(
            self._rate_limit, self._rate_window
        )
        # Reactive blocks are scoped per key when callers identify which key
        # failed. Without a key id, fall back to a single legacy global slot so
        # tests and ad-hoc callers keep their prior provider-wide semantics.
        self._blocked_until_by_key: dict[str, float] = {}
        self._blocked_until_global: float = 0.0
        self._active_key_id: str | None = None
        self._concurrency_sem = asyncio.Semaphore(max_concurrency)
        logger.info(
            "ProviderRateLimiter initialized "
            f"({rate_limit} req / {rate_window}s, max_concurrency={max_concurrency})"
        )

    async def wait_if_blocked(self) -> bool:
        """
        Wait if currently rate limited or throttle to meet quota.

        Returns:
            True if was reactively blocked and waited, False otherwise.
        """
        # A reactive deadline can be installed or extended while this task waits
        # for proactive capacity. Commit the proactive timestamp only if that
        # deadline is still clear, so retries neither burst nor consume unused quota.
        waited_reactively = False
        while True:
            waited_reactively = (
                await self._wait_for_reactive_block() or waited_reactively
            )
            if await self._proactive_limiter.acquire_if(lambda: not self.is_blocked()):
                return waited_reactively

    async def _wait_for_reactive_block(self) -> bool:
        waited = False
        while (wait_time := self.remaining_wait()) > 0:
            logger.warning(
                "Provider rate limit active (reactive), waiting {:.1f}s...",
                wait_time,
            )
            await asyncio.sleep(wait_time)
            waited = True
        return waited

    def extend_reactive_block(
        self, seconds: float, *, key_id: str | None = None
    ) -> None:
        """
        Extend this provider's reactive block by at least ``seconds`` from now.

        When ``key_id`` is provided, the block is scoped to that key only:
        requests that route through a different key are unaffected, which lets
        a healthy replacement key serve traffic immediately after a 429.

        Args:
            seconds: Positive minimum duration for the resulting block.
            key_id: Optional identifier for the API key that hit the limit.
                When omitted, falls back to the active key (or the legacy
                global slot when no active key has been set).
        """
        if seconds <= 0:
            raise ValueError("reactive block duration must be > 0")
        now = time.monotonic()
        deadline = now + seconds
        effective_key = key_id if key_id is not None else self._active_key_id
        if effective_key is None:
            self._blocked_until_global = max(self._blocked_until_global, deadline)
            logger.warning(
                "Provider rate limit set for {:.1f}s (reactive, global)",
                max(0.0, self._blocked_until_global - now),
            )
            return
        previous = self._blocked_until_by_key.get(effective_key, 0.0)
        self._blocked_until_by_key[effective_key] = max(previous, deadline)
        logger.warning(
            "Provider rate limit set for {:.1f}s (reactive, key={})",
            max(0.0, self._blocked_until_by_key[effective_key] - now),
            effective_key,
        )

    def set_active_key_id(self, key_id: str | None) -> None:
        """Identify the API key that the next attempt will use.

        Lets ``is_blocked`` and ``remaining_wait`` answer the question
        "is the active key blocked?" without requiring callers to thread
        the key id through every ``wait_if_blocked`` call.
        """
        self._active_key_id = key_id

    def expire_stale_blocks(self) -> None:
        """Drop per-key reactive blocks that already elapsed under
        ``time.monotonic``. Keeps the bookkeeping map from growing without
        bound under providers with very large key pools."""
        now = time.monotonic()
        stale = [
            key
            for key, deadline in self._blocked_until_by_key.items()
            if deadline <= now
        ]
        for key in stale:
            self._blocked_until_by_key.pop(key, None)

    def is_blocked(self) -> bool:
        """Check if the currently active key (or global) is reactively blocked."""
        return self.is_blocked_for_key(self._active_key_id)

    def is_blocked_for_key(self, key_id: str | None) -> bool:
        """Check whether this provider is blocked for ``key_id``."""
        now = time.monotonic()
        if now < self._blocked_until_global:
            return True
        return key_id is not None and now < self._blocked_until_by_key.get(key_id, 0.0)

    def remaining_wait(self) -> float:
        """Get remaining reactive wait time for the active key in seconds."""
        return self.remaining_wait_for_key(self._active_key_id)

    def remaining_wait_for_key(self, key_id: str | None) -> float:
        """Get remaining reactive wait time for ``key_id`` in seconds."""
        now = time.monotonic()
        deadline = self._blocked_until_global
        if key_id is not None:
            candidate = self._blocked_until_by_key.get(key_id, 0.0)
            if candidate > deadline:
                deadline = candidate
        return max(0.0, deadline - now)

    @asynccontextmanager
    async def concurrency_slot(self) -> AsyncIterator[None]:
        """Async context manager that holds one concurrency slot for a stream.

        Blocks until a slot is available (controlled by max_concurrency).
        """
        await self._concurrency_sem.acquire()
        try:
            yield
        finally:
            self._concurrency_sem.release()

    async def execute_with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        provider_failure_override: ProviderFailureOverride | None = None,
        max_retries: int = DEFAULT_UPSTREAM_MAX_RETRIES,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        jitter: float = 1.0,
        on_key_failure: KeyFailureCallback | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute an async callable with rate limiting and retry on transient limits.

        Waits for the proactive limiter before each attempt. On ``429`` (rate limit)
        or upstream ``5xx`` server errors, applies exponential backoff with jitter
        and sets the reactive block before retrying. Pre-response transport errors
        use the same attempt budget and backoff schedule without setting the
        reactive provider block.

        When *on_key_failure* is provided, it is called before the retry delay
        on authentication (401) and rate-limit (429) failures so the caller can
        rotate to the next API key. Auth failures count toward the attempt budget
        when a key-failure hook is installed (otherwise they are non-retryable).

        Args:
            fn: Async callable to execute.
            provider_failure_override: Optional provider-specific semantic
                classifier applied before shared retry qualification.
            max_retries: Maximum number of retry attempts after the first failure.
            base_delay: Base delay in seconds for exponential backoff.
            max_delay: Maximum delay cap in seconds.
            jitter: Maximum random jitter in seconds added to each delay.
            on_key_failure: Optional callback invoked with (FailureKind, cooldown_s)
                before retrying on 401 or 429 so the caller can rotate keys.

        Returns:
            The result of the callable.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exc: Exception | None = None
        total_attempts = 1 + max_retries

        for attempt in range(total_attempts):
            await self.wait_if_blocked()
            # Capture the active key id *before* the attempt because the
            # key-rotation hook may run mid-flight and change _active_key_id.
            attempt_key_id = self._active_key_id

            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                effective_error = (
                    provider_failure_override(e)
                    if provider_failure_override is not None
                    else None
                )
                if effective_error is None:
                    effective_error = e

                # Check for key-failure hook: auth (401/403) and rate-limit (429)
                key_failure_kind: FailureKind | None = None
                if on_key_failure is not None:
                    if isinstance(effective_error, ExecutionFailure):
                        if effective_error.kind == FailureKind.AUTHENTICATION:
                            key_failure_kind = FailureKind.AUTHENTICATION
                    elif isinstance(
                        e,
                        (
                            openai.AuthenticationError,
                            httpx.HTTPStatusError,
                        ),
                    ):
                        status = getattr(
                            getattr(e, "response", None), "status_code", None
                        ) or getattr(e, "status_code", None)
                        if status in (401, 403):
                            key_failure_kind = FailureKind.AUTHENTICATION

                status = retryable_upstream_status(effective_error)
                transport_error = status is None and retryable_upstream_transport_error(
                    effective_error
                )

                # Non-retryable — unless key-failure hook handles it
                if status is None and not transport_error:
                    if key_failure_kind is not None and on_key_failure is not None:
                        on_key_failure(key_failure_kind, 0.0)
                        last_exc = e
                        if attempt < max_retries:
                            delay = min(base_delay * (2**attempt), max_delay)
                            delay += random.uniform(0, jitter)
                            attempt_no = attempt + 1
                            logger.warning(
                                "Auth failure (401/403), key rotated, "
                                "attempt {}/{}. Retrying in {:.1f}s...",
                                attempt_no,
                                total_attempts,
                                delay,
                            )
                            await asyncio.sleep(delay)
                            continue
                        break
                    raise

                if status is None:
                    label = f"Provider transport error ({type(e).__name__})"
                else:
                    label = (
                        "Rate limited (429)"
                        if status == 429
                        else f"Upstream server error ({status})"
                    )

                # Notify key-failure hook on 429 so caller can rotate keys.
                # Honor upstream Retry-After / X-RateLimit-Reset when present
                # so a single bad key does not eat 60s of system-wide recovery.
                if status == 429 and on_key_failure is not None:
                    on_key_failure(
                        FailureKind.RATE_LIMIT,
                        cooldown_seconds_from_exception(effective_error),
                    )

                last_exc = e
                if attempt >= max_retries:
                    logger.warning(
                        "{} retry exhausted after {} retries (attempts={})",
                        label,
                        max_retries,
                        total_attempts,
                    )
                    break

                delay = min(base_delay * (2**attempt), max_delay)
                delay += random.uniform(0, jitter)
                attempt_no = attempt + 1
                logger.warning(
                    "{}, attempt {}/{}. Retrying in {:.1f}s...",
                    label,
                    attempt_no,
                    total_attempts,
                    delay,
                )
                trace_event(
                    stage="provider",
                    event="provider.retry.scheduled",
                    source="provider",
                    status_code=status,
                    exc_type=type(e).__name__,
                    attempt=attempt_no,
                    max_attempts=total_attempts,
                    delay_s=round(delay, 3),
                )
                if status is not None:
                    self.extend_reactive_block(delay, key_id=attempt_key_id)
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc
