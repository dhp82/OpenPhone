"""Screen keepalive for PhoneClaw.

Strategy
--------
Primary — ``idleTimerDisabled`` (no touch required)
    WDA exposes ``POST /wda/settings`` which maps directly to Apple's
    ``[[UIApplication sharedApplication] setIdleTimerDisabled:YES]``.
    Setting this flag tells iOS to never engage the auto-lock idle timer for
    the duration of the WDA session, without touching any UI element at all.

    A background thread runs ``GET /status`` every *interval* seconds purely
    to keep the WDA HTTP session from timing out (some proxies/firewalls drop
    idle TCP connections after 30–60 s).

Fallback — periodic touch (when ``idleTimerDisabled`` is not supported)
    Older / custom WDA builds may not expose the ``idleTimerDisabled``
    setting.  In that case the keepalive falls back to sending a synthetic
    touch event every *interval* seconds via the W3C Actions API.

    Touch target: the **horizontal centre, vertical middle** of the screen
    (x = 50 %, y = 50 %).  The middle of the screen is the one area that
    is most reliably neutral across all apps — it avoids:
    - Status bar (scroll-to-top trigger)
    - Dynamic Island / notch (live-activity expansion)
    - Bottom home-indicator strip (may be interactive in some apps)
    - Left/right edges (back-swipe gesture zones)
    The downside is that in rare cases it may tap on a button in the current
    app; this is acceptable because the fallback is only used when the cleaner
    API is unavailable.

Usage::

    keepalive = ScreenKeepalive(wda_url="http://localhost:8100",
                                session_id="...",
                                interval=25)
    keepalive.start()
    ...                 # runs in background while your main code executes
    keepalive.stop()    # or just let the process exit (daemon thread auto-dies)
"""

import threading
from typing import Optional

import requests


class ScreenKeepalive:
    """
    Prevents the iOS device screen from sleeping during an interactive session.

    On ``start()``:
      1. Sends ``POST /wda/settings {"settings": {"idleTimerDisabled": true}}``
         to disable iOS auto-lock at the system level — no touches needed.
      2. Starts a lightweight daemon thread that pings ``GET /status`` every
         *interval* seconds to keep the WDA HTTP session alive.

    On ``stop()``:
      1. Re-enables the idle timer via ``idleTimerDisabled: false``.
      2. Stops the background thread.

    If the ``idleTimerDisabled`` setting is not supported by the WDA build, a
    warning is printed and the keepalive falls back to a periodic synthetic
    touch event (see module docstring).
    """

    def __init__(
        self,
        wda_url: str = "http://localhost:8100",
        session_id: Optional[str] = None,
        interval: float = 25.0,
        verbose: bool = True,
    ):
        """
        Args:
            wda_url:    WebDriverAgent base URL.
            session_id: WDA session ID (required).
            interval:   Seconds between WDA heartbeat pings / fallback taps.
                        Should be shorter than the device auto-lock timeout.
            verbose:    Print a brief log line on start/stop.
        """
        self.wda_url = wda_url.rstrip("/")
        self.session_id = session_id
        self.interval = interval
        self.verbose = verbose

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # True when idleTimerDisabled was successfully set — so we know to
        # clear it on stop().
        self._idle_timer_disabled = False
        # True when the primary API is unavailable and we use touch fallback.
        self._using_touch_fallback = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Enable screen keepalive and start the background heartbeat thread."""
        if self._thread and self._thread.is_alive():
            return

        self._idle_timer_disabled = self._set_idle_timer_disabled(True)

        if self._idle_timer_disabled:
            if self.verbose:
                print(
                    f"[Keepalive] idleTimerDisabled=true — screen will stay on "
                    f"without touch events.  Heartbeat every {self.interval}s."
                )
            loop_target = self._heartbeat_loop
        else:
            # idleTimerDisabled not supported by this WDA build — fall back to
            # a periodic touch event.
            self._using_touch_fallback = True
            if self.verbose:
                print(
                    f"[Keepalive] idleTimerDisabled not supported — falling back "
                    f"to touch keepalive every {self.interval}s."
                )
            loop_target = self._touch_loop

        self._stop.clear()
        self._thread = threading.Thread(
            target=loop_target,
            name="ScreenKeepalive",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop keepalive and restore the idle timer."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.interval + 2)

        if self._idle_timer_disabled:
            self._set_idle_timer_disabled(False)
            self._idle_timer_disabled = False

        if self.verbose:
            print("[Keepalive] Stopped.")

    # ------------------------------------------------------------------
    # Primary: system-level idle timer control
    # ------------------------------------------------------------------

    def _set_idle_timer_disabled(self, disabled: bool) -> bool:
        """
        Call ``POST /wda/settings`` to enable or disable the iOS idle timer.

        Returns True on success, False if the setting is unsupported or the
        request fails.
        """
        url = f"{self.wda_url}/wda/settings"
        try:
            r = requests.post(
                url,
                json={"settings": {"idleTimerDisabled": disabled}},
                timeout=8,
            )
            if r.ok:
                return True
            # WDA returns 400/500 when the setting is unknown
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Primary loop: lightweight WDA heartbeat (no UI interaction)
    # ------------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        """Ping ``GET /status`` to keep the WDA TCP session alive."""
        while not self._stop.wait(self.interval):
            try:
                requests.get(f"{self.wda_url}/status", timeout=8)
            except Exception as exc:
                if self.verbose:
                    print(f"[Keepalive] Heartbeat warning: {exc}")

    # ------------------------------------------------------------------
    # Fallback loop: synthetic touch event
    # ------------------------------------------------------------------

    def _touch_loop(self) -> None:
        """Send a synthetic touch to the screen centre every *interval* seconds."""
        while not self._stop.wait(self.interval):
            try:
                self._touch_centre()
            except Exception as exc:
                if self.verbose:
                    print(f"[Keepalive] Touch warning: {exc}")

    def _touch_centre(self) -> None:
        """
        Send a single synthetic tap to the centre of the screen (50 %, 50 %).

        The screen centre is chosen as the least-disruptive touch position
        available as a fallback: it avoids the status bar (scroll-to-top),
        Dynamic Island / notch, home-indicator strip, and edge-swipe zones.
        """
        w, h = self._get_logical_size()
        x = w // 2
        y = h // 2

        url = f"{self.wda_url}/session/{self.session_id}/actions"
        payload = {
            "actions": [
                {
                    "type": "pointer",
                    "id": "keepalive_finger",
                    "parameters": {"pointerType": "touch"},
                    "actions": [
                        {"type": "pointerMove", "duration": 0, "x": x, "y": y},
                        {"type": "pointerDown", "button": 0},
                        {"type": "pause",       "duration": 50},
                        {"type": "pointerUp",   "button": 0},
                    ],
                }
            ]
        }
        r = requests.post(url, json=payload, timeout=8)
        if self.verbose:
            status = "ok" if r.ok else f"HTTP {r.status_code}"
            print(f"[Keepalive] Touch centre ({x}, {y}) — {status}")

    def _get_logical_size(self) -> tuple[int, int]:
        """Return logical screen dimensions, queried once and cached."""
        if hasattr(self, "_logi_w") and self._logi_w:
            return self._logi_w, self._logi_h  # type: ignore[return-value]

        try:
            url = f"{self.wda_url}/session/{self.session_id}/window/size"
            r = requests.get(url, timeout=5)
            if r.ok:
                v = r.json().get("value", {})
                self._logi_w = int(v.get("width",  393))
                self._logi_h = int(v.get("height", 852))
                return self._logi_w, self._logi_h
        except Exception:
            pass

        self._logi_w, self._logi_h = 393, 852
        return self._logi_w, self._logi_h
