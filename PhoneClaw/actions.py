"""Action execution for iOS devices via WebDriverAgent."""

import time
from typing import Optional, Tuple

# iOS app bundle IDs
# Add entries here to extend launch() support.
# Find bundle IDs via: ideviceinstaller -l  or  GET /wda/apps/list
APP_PACKAGES_IOS = {
    # ── Apple system apps ──────────────────────────────────────────────
    "Safari":           "com.apple.mobilesafari",
    "Settings":         "com.apple.Preferences",
    "Messages":         "com.apple.MobileSMS",
    "Mail":             "com.apple.mobilemail",
    "Photos":           "com.apple.mobileslideshow",
    "Camera":           "com.apple.camera",
    "Clock":            "com.apple.mobiletimer",
    "Calendar":         "com.apple.mobilecal",
    "Maps":             "com.apple.Maps",
    "Music":            "com.apple.Music",
    "App Store":        "com.apple.AppStore",
    "Notes":            "com.apple.mobilenotes",
    "Reminders":        "com.apple.reminders",
    "Weather":          "com.apple.weather",
    "Calculator":       "com.apple.calculator",
    "Contacts":         "com.apple.MobileAddressBook",
    "FaceTime":         "com.apple.facetime",
    "Phone":            "com.apple.mobilephone",
    "Health":           "com.apple.Health",
    "Wallet":           "com.apple.Passbook",
    "Files":            "com.apple.DocumentsApp",
    "Podcasts":         "com.apple.podcasts",
    "Shortcuts":        "com.apple.shortcuts",
    "Translate":        "com.apple.Translate",
    "Find My":          "com.apple.findmy",

    # ── Social / communication ─────────────────────────────────────────
    "WeChat":           "com.tencent.xin",       # NOT com.tencent.xinWeChat
    "QQ":               "com.tencent.mqq",
    "Weibo":            "com.sina.weibo",
    "Feishu":           "com.bytedance.feishu",
    "Lark":             "com.bytedance.lark",
    "DingTalk":         "com.laiwang.DingTalk",
    "钉钉":             "com.laiwang.DingTalk",

    # ── Shopping / delivery ────────────────────────────────────────────
    # com.meituan.imeituan is the current "美团" app (food delivery + all
    # services). The old super-app com.sankuai.meituan is rarely installed.
    "Meituan":          "com.meituan.imeituan",
    "美团":             "com.meituan.imeituan",
    "Meituan Waimai":   "com.meituan.imeituan",
    "美团外卖":         "com.meituan.imeituan",
    "Taobao":           "com.taobao.taobao4iphone",
    "淘宝":             "com.taobao.taobao4iphone",
    "JD":               "com.jingdong.app.mall",
    "京东":             "com.jingdong.app.mall",
    "Pinduoduo":        "com.xunmeng.pinduoduo",
    "拼多多":           "com.xunmeng.pinduoduo",
    "Xiaohongshu":      "com.xingin.discover",
    "小红书":           "com.xingin.discover",
    "Eleme":            "me.ele.ios",
    "饿了么":           "me.ele.ios",

    # ── Finance / payment ─────────────────────────────────────────────
    "Alipay":           "com.alipay.iphoneclient",
    "支付宝":           "com.alipay.iphoneclient",

    # ── Travel / maps ──────────────────────────────────────────────────
    "Didi":             "com.xiaojukeji.didi.passenger.activity",
    "滴滴":             "com.xiaojukeji.didi.passenger.activity",
    "Ctrip":            "com.ctrip.inner.wireless",
    "携程":             "com.ctrip.inner.wireless",
    "Gaode Maps":       "com.autonavi.amap",
    "高德地图":         "com.autonavi.amap",
    "Baidu Maps":       "com.baidu.map",
    "百度地图":         "com.baidu.map",

    # ── Video / streaming ─────────────────────────────────────────────
    "Douyin":           "com.ss.iphone.ugc.Aweme",
    "抖音":             "com.ss.iphone.ugc.Aweme",
    "Bilibili":         "tv.danmaku.bilianime",
    "哔哩哔哩":         "tv.danmaku.bilianime",
    "iQIYI":            "com.qiyi.iphone",
    "爱奇艺":           "com.qiyi.iphone",
    "Youku":            "com.youku.YouKu",
    "优酷":             "com.youku.YouKu",
    "Tencent Video":    "com.tencent.now",
    "腾讯视频":         "com.tencent.now",

    # ── Music ──────────────────────────────────────────────────────────
    "NetEase Music":    "com.netease.cloudmusic",
    "网易云音乐":       "com.netease.cloudmusic",
    "QQ Music":         "com.tencent.qqmusic",
    "QQ音乐":           "com.tencent.qqmusic",

    # ── Knowledge / tools ─────────────────────────────────────────────
    "Zhihu":            "com.zhihu.ios",
    "知乎":             "com.zhihu.ios",
    "Baidu":            "com.baidu.BaiduMobile",
    "百度":             "com.baidu.BaiduMobile",

    # ── Google apps ───────────────────────────────────────────────────
    "Gmail":            "com.google.Gmail",         # capital G
    "Google Maps":      "com.google.Maps",
    "Google Chrome":    "com.google.chrome.ios",
    "Chrome":           "com.google.chrome.ios",
    "YouTube":          "com.google.ios.youtube",
}

SCALE_FACTOR = 3  # 3 for most modern iPhone


def _physical_to_logical(x: int, y: int) -> Tuple[int, int]:
    """Convert physical coordinates (screenshot) to logical coordinates (WDA)."""
    return int(x / SCALE_FACTOR), int(y / SCALE_FACTOR)


def _logical_to_physical(x: int, y: int) -> Tuple[int, int]:
    """Convert logical coordinates (WDA) to physical coordinates (screenshot)."""
    return int(x * SCALE_FACTOR), int(y * SCALE_FACTOR)


def _get_wda_session_url(wda_url: str, session_id: Optional[str], endpoint: str) -> str:
    """Get the correct WDA URL for a session endpoint."""
    base = wda_url.rstrip("/")
    if session_id:
        return f"{base}/session/{session_id}/{endpoint}"
    else:
        return f"{base}/{endpoint}"


class IOSActionHandler:
    """Handles execution of actions for iOS devices."""

    def __init__(
        self,
        wda_url: str = "http://localhost:8100",
        session_id: Optional[str] = None,
    ):
        self.wda_url = wda_url
        self.session_id = session_id

    def tap(self, x: int, y: int, delay: float = 1.0) -> bool:
        """Tap at the specified coordinates."""
        try:
            import requests
            url = _get_wda_session_url(self.wda_url, self.session_id, "actions")

            actions = {
                "actions": [
                    {
                        "type": "pointer",
                        "id": "finger1",
                        "parameters": {"pointerType": "touch"},
                        "actions": [
                            {"type": "pointerMove", "duration": 0, "x": x / SCALE_FACTOR, "y": y / SCALE_FACTOR},
                            {"type": "pointerDown", "button": 0},
                            {"type": "pause", "duration": 100},
                            {"type": "pointerUp", "button": 0},
                        ],
                    }
                ]
            }

            response = requests.post(url, json=actions, timeout=15, verify=False)
            time.sleep(delay)
            return response.status_code in (200, 201)
        except Exception as e:
            print(f"Error tapping: {e}")
            return False

    def double_tap(self, x: int, y: int, delay: float = 1.0) -> bool:
        """Double tap at the specified coordinates."""
        try:
            import requests
            url = _get_wda_session_url(self.wda_url, self.session_id, "actions")

            actions = {
                "actions": [
                    {
                        "type": "pointer",
                        "id": "finger1",
                        "parameters": {"pointerType": "touch"},
                        "actions": [
                            {"type": "pointerMove", "duration": 0, "x": x / SCALE_FACTOR, "y": y / SCALE_FACTOR},
                            {"type": "pointerDown", "button": 0},
                            {"type": "pause", "duration": 100},
                            {"type": "pointerUp", "button": 0},
                            {"type": "pause", "duration": 100},
                            {"type": "pointerDown", "button": 0},
                            {"type": "pause", "duration": 100},
                            {"type": "pointerUp", "button": 0},
                        ],
                    }
                ]
            }

            response = requests.post(url, json=actions, timeout=10, verify=False)
            time.sleep(delay)
            return response.status_code in (200, 201)
        except Exception as e:
            print(f"Error double tapping: {e}")
            return False

    def long_press(self, x: int, y: int, duration: float = 3.0, delay: float = 1.0) -> bool:
        """Long press at the specified coordinates."""
        try:
            import requests
            url = _get_wda_session_url(self.wda_url, self.session_id, "actions")

            duration_ms = int(duration * 1000)
            actions = {
                "actions": [
                    {
                        "type": "pointer",
                        "id": "finger1",
                        "parameters": {"pointerType": "touch"},
                        "actions": [
                            {"type": "pointerMove", "duration": 0, "x": x / SCALE_FACTOR, "y": y / SCALE_FACTOR},
                            {"type": "pointerDown", "button": 0},
                            {"type": "pause", "duration": duration_ms},
                            {"type": "pointerUp", "button": 0},
                        ],
                    }
                ]
            }

            response = requests.post(url, json=actions, timeout=int(duration + 10), verify=False)
            time.sleep(delay)
            return response.status_code in (200, 201)
        except Exception as e:
            print(f"Error long pressing: {e}")
            return False

    def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: Optional[float] = None,
        delay: float = 1.0,
    ) -> bool:
        """Swipe from start to end coordinates."""
        try:
            import requests

            if duration is None:
                dist_sq = (start_x - end_x) ** 2 + (start_y - end_y) ** 2
                duration = dist_sq / 1000000
                duration = max(0.3, min(duration, 2.0))

            url = _get_wda_session_url(self.wda_url, self.session_id, "wda/dragfromtoforduration")

            payload = {
                "fromX": start_x / SCALE_FACTOR,
                "fromY": start_y / SCALE_FACTOR,
                "toX": end_x / SCALE_FACTOR,
                "toY": end_y / SCALE_FACTOR,
                "duration": duration,
            }

            response = requests.post(url, json=payload, timeout=int(duration + 10), verify=False)
            time.sleep(delay)
            return response.status_code in (200, 201)
        except Exception as e:
            print(f"Error swiping: {e}")
            return False

    def back(self, delay: float = 1.0) -> bool:
        """Navigate back (swipe from left edge)."""
        try:
            import requests
            url = _get_wda_session_url(self.wda_url, self.session_id, "wda/dragfromtoforduration")

            screen_width, screen_height = self.get_screen_size()

            from_x = 0
            from_y = screen_height // 2
            to_x = screen_width // 3
            to_y = from_y

            payload = {
                "fromX": from_x,
                "fromY": from_y,
                "toX": to_x,
                "toY": to_y,
                "duration": 0.3,
            }

            response = requests.post(url, json=payload, timeout=10, verify=False)
            time.sleep(delay)
            return response.status_code in (200, 201)
        except Exception as e:
            print(f"Error performing back gesture: {e}")
            return False

    def home(self, delay: float = 1.0) -> bool:
        """Press the home button."""
        try:
            import requests
            url = f"{self.wda_url.rstrip('/')}/wda/homescreen"
            response = requests.post(url, timeout=10, verify=False)
            time.sleep(delay)
            return response.status_code in (200, 201)
        except Exception as e:
            print(f"Error pressing home: {e}")
            return False

    def launch_app(self, app_name: str, delay: float = 2.0) -> bool:
        """
        Launch (or bring to foreground) an app by name via WDA.

        Uses /wda/apps/activate rather than /wda/apps/launch:
        - activate  → iOS system-level "open application" (works for ALL installed apps)
        - launch    → XCTest XCUIApplication cold-start (fails for third-party apps on
                      real devices with "FBSApplicationLibrary returned nil" error)
        """
        if app_name not in APP_PACKAGES_IOS:
            print(f"App '{app_name}' not found in APP_PACKAGES_IOS")
            return False

        try:
            import requests
            bundle_id = APP_PACKAGES_IOS[app_name]
            url = _get_wda_session_url(self.wda_url, self.session_id, "wda/apps/activate")

            response = requests.post(
                url, json={"bundleId": bundle_id}, timeout=15, verify=False
            )

            time.sleep(delay)
            # activate returns null value on success (status 200)
            if response.status_code in (200, 201):
                data = response.json()
                value = data.get("value")
                if isinstance(value, dict) and "error" in value:
                    msg = value.get("message", "")
                    if "NotFound" in msg or "returned nil" in msg:
                        print(f"[launch] '{app_name}' not installed on this device.")
                    else:
                        print(f"[launch] activate error: {msg[:120]}")
                    return False
                return True
            elif response.status_code == 400:
                # HTTP 400 also means the app is not installed on this device
                print(f"[launch] '{app_name}' ({bundle_id}) is not installed on this device.")
                return False
            else:
                print(f"[launch] Unexpected HTTP {response.status_code} for '{app_name}'")
                return False
        except Exception as e:
            print(f"Error launching app: {e}")
            return False

    def type_text(self, text: str, frequency: int = 60) -> bool:
        """Type text into the currently focused input field."""
        try:
            import requests
            url = _get_wda_session_url(self.wda_url, self.session_id, "wda/keys")

            response = requests.post(
                url, json={"value": list(text), "frequency": frequency}, timeout=30, verify=False
            )

            return response.status_code in (200, 201)
        except Exception as e:
            print(f"Error typing text: {e}")
            return False

    def clear_text(self) -> bool:
        """Clear text in the currently focused input field."""
        try:
            import requests
            url = _get_wda_session_url(self.wda_url, self.session_id, "element/active")

            response = requests.get(url, timeout=10, verify=False)

            if response.status_code == 200:
                data = response.json()
                element_id = data.get("value", {}).get("ELEMENT") or data.get("value", {}).get("element-6066-11e4-a52e-4f735466cecf")

                if element_id:
                    clear_url = _get_wda_session_url(self.wda_url, self.session_id, f"element/{element_id}/clear")
                    response = requests.post(clear_url, timeout=10, verify=False)
                    return response.status_code in (200, 201)

            return False
        except Exception as e:
            print(f"Error clearing text: {e}")
            return False

    def hide_keyboard(self) -> bool:
        """Hide the on-screen keyboard."""
        try:
            import requests
            url = f"{self.wda_url.rstrip('/')}/wda/keyboard/dismiss"
            response = requests.post(url, timeout=10, verify=False)
            return response.status_code in (200, 201)
        except Exception as e:
            print(f"Error hiding keyboard: {e}")
            return False

    def get_current_app(self) -> str:
        """Get the currently active app name."""
        try:
            import requests
            response = requests.get(
                f"{self.wda_url.rstrip('/')}/wda/activeAppInfo", timeout=5, verify=False
            )

            if response.status_code == 200:
                data = response.json()
                value = data.get("value", {})
                bundle_id = value.get("bundleId", "")

                if bundle_id:
                    for app_name, package in APP_PACKAGES_IOS.items():
                        if package == bundle_id:
                            return app_name

                return "System Home"

        except Exception as e:
            print(f"Error getting current app: {e}")

        return "System Home"

    def get_screen_size(self) -> tuple[int, int]:
        """Get the screen dimensions."""
        try:
            import requests
            url = _get_wda_session_url(self.wda_url, self.session_id, "window/size")

            response = requests.get(url, timeout=5, verify=False)

            if response.status_code == 200:
                data = response.json()
                value = data.get("value", {})
                width = value.get("width", 375)
                height = value.get("height", 812)
                return width, height

        except Exception as e:
            print(f"Error getting screen size: {e}")

        return 375, 812
