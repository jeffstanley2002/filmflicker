"""Small process-local rate limiter for a single-worker portfolio deployment."""
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

_events: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def rate_limit(scope: str, limit: int, window_seconds: int):
    def dependency(request: Request) -> None:
        forwarded = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        client = forwarded or (request.client.host if request.client else "unknown")
        key = f"{scope}:{client}"
        now = time.monotonic()
        cutoff = now - window_seconds
        with _lock:
            if len(_events) > 10_000:
                stale = [name for name, events in _events.items() if not events or events[-1] <= cutoff]
                for name in stale:
                    _events.pop(name, None)
            bucket = _events[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = max(1, int(window_seconds - (now - bucket[0])))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests. Please try again shortly.",
                    headers={"Retry-After": str(retry_after)},
                )
            bucket.append(now)

    return dependency
