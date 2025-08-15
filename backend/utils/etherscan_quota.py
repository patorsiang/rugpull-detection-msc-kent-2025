from __future__ import annotations

import time
import threading
from typing import Dict, Any, Optional

import requests

from backend.utils.constants import (
    ETHERSCAN_BASE_URL,
    ETHERSCAN_API_KEY,
    ETHERSCAN_HTTP_TIMEOUT,
    ETHERSCAN_QUOTA_TTL,
    ETHERSCAN_DAILY_LIMIT,
)

# ---- public API -------------------------------------------------------------

class QuotaExceeded(RuntimeError):
    """Raised when an operation would exceed the daily Etherscan quota."""


class EtherscanQuotaGuard:
    """
    Threadsafe snapshot of Etherscan's request quota.

    - Calls v2 endpoint: module=getapilimit&action=getapilimit
    - Parses new keys: creditsUsed, creditsAvailable, creditLimit, limitInterval, intervalExpiryTimespan
    - Falls back to older spellings if present
    - Soft-updates remaining via count() to avoid over-fetching between polls
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._base_url: str = ETHERSCAN_BASE_URL
        self._apikey: str = ETHERSCAN_API_KEY
        self._timeout: int = int(ETHERSCAN_HTTP_TIMEOUT)
        self._ttl: int = int(ETHERSCAN_QUOTA_TTL)

        # state
        self._daily_limit: int = int(ETHERSCAN_DAILY_LIMIT)
        self._daily_used: int = 0
        self._daily_remaining: int = self._daily_limit
        self._limit_interval: Optional[str] = None                  # e.g., "daily"
        self._expiry_timespan_raw: Optional[str] = None             # e.g., "07:20:05"
        self._reset_eta_seconds: Optional[int] = None               # parsed from timespan when available
        self._last_update: float = 0.0

        # optional: stash raw payload for diagnostics
        self._last_raw: Optional[Dict[str, Any]] = None

        # one eager refresh (non-fatal)
        try:
            self._refresh()
        except Exception:
            pass

    # ---- public helpers -----------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return current view; refresh if stale."""
        with self._lock:
            self._refresh_if_stale_locked()
            return {
                "daily_limit": int(self._daily_limit),
                "daily_used": int(self._daily_used),
                "daily_remaining": int(self._daily_remaining),
                "limit_interval": self._limit_interval,
                "interval_expiry_timespan": self._expiry_timespan_raw,
                "reset_eta_seconds": self._reset_eta_seconds,
                "last_update": self._last_update,
            }

    def is_exhausted(self, require_calls: int = 1) -> bool:
        """True if fewer than 'require_calls' remain in this window."""
        if require_calls <= 0:
            return False
        with self._lock:
            self._refresh_if_stale_locked()
            return self._daily_remaining < require_calls

    def require_available_or_raise(self, n: int = 1) -> None:
        """Raise QuotaExceeded if fewer than n calls remain."""
        if self.is_exhausted(n):
            raise QuotaExceeded(
                f"Etherscan quota exhausted (need {n}, remaining {self._daily_remaining})."
            )

    def count(self, n: int = 1) -> None:
        """
        Optimistically decrement remaining (useful immediately after a known batch).
        Not required if you rely only on periodic polling.
        """
        if n <= 0:
            return
        with self._lock:
            self._daily_used = max(0, self._daily_used + n)
            self._daily_remaining = max(0, self._daily_remaining - n)

    # ---- internals ----------------------------------------------------------

    def _refresh_if_stale_locked(self) -> None:
        now = time.time()
        if now - self._last_update >= self._ttl:
            self._refresh_locked()

    def _refresh(self) -> None:
        with self._lock:
            self._refresh_locked()

    def _refresh_locked(self) -> None:
        """
        Hit the quota endpoint and parse leniently.

        v2 example:
        {
          "status":"1",
          "message":"OK",
          "result":{
            "creditsUsed":207,
            "creditsAvailable":499793,
            "creditLimit":500000,
            "limitInterval":"daily",
            "intervalExpiryTimespan":"07:20:05"
          }
        }
        """
        if not self._apikey:
            # no key => keep defaults; don't crash
            self._last_update = time.time()
            return

        params = {
            "module": "getapilimit",
            "action": "getapilimit",
            "apikey": self._apikey,
        }
        try:
            r = requests.get(self._base_url, params=params, timeout=self._timeout)
            if r.status_code == 429:
                # Treat as exhausted until next refresh
                self._daily_remaining = 0
                self._last_update = time.time()
                return

            r.raise_for_status()
            data = r.json()
        except Exception:
            # network problems: keep previous snapshot; just mark refreshed time
            self._last_update = time.time()
            return

        res = data.get("result") or {}
        self._last_raw = res  # keep for diagnostics

        # New v2 keys (first choice), with fallbacks to older spellings
        limit = _first_int(
            res,
            "creditLimit",        # v2
            "DailyCreditLimit",   # legacy variants
            "dailyLimit",
            "DailyLimit",
            "planDailyLimit",
        )
        used = _first_int(
            res,
            "creditsUsed",        # v2
            "UsedDailyCredits",   # legacy
            "dailyUsed",
            "DailyUsed",
        )
        remaining = _first_int(
            res,
            "creditsAvailable",   # v2
            "CreditsRemaining",   # legacy
            "dailyRemaining",
            "DailyRemaining",
        )

        # Derive missing fields safely
        if limit is None:
            limit = self._daily_limit
        if remaining is None and used is not None:
            remaining = max(0, int(limit) - int(used))
        if used is None and remaining is not None:
            used = max(0, int(limit) - int(remaining))
        if remaining is None:
            remaining = max(0, int(limit) - int(used or 0))

        # Interval metadata (best-effort)
        self._limit_interval = _first_str(res, "limitInterval")  # e.g. "daily"
        self._expiry_timespan_raw = _first_str(res, "intervalExpiryTimespan")  # e.g. "07:20:05"
        self._reset_eta_seconds = _parse_hms(self._expiry_timespan_raw) if self._expiry_timespan_raw else None

        # Commit snapshot
        self._daily_limit = int(limit)
        self._daily_used = int(used or 0)
        self._daily_remaining = int(remaining)
        self._last_update = time.time()


def _first_int(d: Dict[str, Any], *keys: str) -> Optional[int]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return int(d[k])
            except Exception:
                try:
                    return int(str(d[k]).strip())
                except Exception:
                    continue
    return None


def _first_str(d: Dict[str, Any], *keys: str) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            return str(d[k])
    return None


def _parse_hms(hms: str) -> Optional[int]:
    """
    Parse 'HH:MM:SS' -> total seconds. Return None on failure.
    """
    try:
        parts = str(hms).split(":")
        if len(parts) != 3:
            return None
        h, m, s = (int(parts[0]), int(parts[1]), int(parts[2]))
        return h * 3600 + m * 60 + s
    except Exception:
        return None


# global singleton used throughout the app
quota_guard = EtherscanQuotaGuard()
