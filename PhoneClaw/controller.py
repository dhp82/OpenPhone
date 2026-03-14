"""iOS Controller - adapts iOS device control to Android-Lab's controller interface."""

import os
import time
from typing import Tuple, Optional

from PhoneClaw.actions import IOSActionHandler, _physical_to_logical, SCALE_FACTOR
from PhoneClaw.screenshot import get_screenshot, save_screenshot, Screenshot
from PhoneClaw.hierarchy import get_page_source


class IOSController:
    """
    iOS Controller that adapts iOS device control to Android-Lab's controller interface.

    Provides methods compatible with Android-Lab's AndroidController,
    allowing iOS devices to be used with the same agent code.
    """

    def __init__(self, wda_url: str = "http://localhost:8100", session_id: Optional[str] = None):
        self.action_handler = IOSActionHandler(wda_url=wda_url, session_id=session_id)
        self.wda_url = wda_url
        self.session_id = session_id
        self.width, self.height = self.get_device_size()
        self.viewport_size = (self.width, self.height)
        # Default screenshot dir; overridden per-task by run_phoneclaw.py.
        # Directory is NOT pre-created here to avoid leaving empty folders.
        self.screenshot_dir = "./phoneclaw_screenshots"

    def get_device_size(self) -> Tuple[int, int]:
        """Get device screen size."""
        return self.action_handler.get_screen_size()

    def get_current_activity(self) -> str:
        """Get current app name (iOS equivalent of Android activity)."""
        return self.action_handler.get_current_app()

    def get_current_app(self) -> str:
        """Alias for get_current_activity for compatibility."""
        return self.get_current_activity()

    def tap(self, x: int, y: int) -> bool:
        """Tap at coordinates."""
        return self.action_handler.tap(x, y)

    def text(self, input_str: str) -> bool:
        """Type text into focused input field."""
        self.action_handler.clear_text()
        time.sleep(0.5)
        success = self.action_handler.type_text(input_str)
        time.sleep(0.5)
        self.action_handler.hide_keyboard()
        return success

    def long_press(self, x: int, y: int, duration: int = 3000) -> bool:
        """Long press at coordinates. duration in milliseconds."""
        return self.action_handler.long_press(x, y, duration=duration / 1000.0)

    def swipe(self, x: int, y: int, direction: str, dist: str = "medium", quick: bool = False) -> bool:
        """Swipe from coordinates in specified direction."""
        x_logical, y_logical = _physical_to_logical(x, y)

        dist_multiplier = {"short": 0.3, "medium": 0.5, "long": 0.7}.get(dist, 0.5)

        if direction == "up":
            end_x_logical = x_logical
            end_y_logical = max(0, int(y_logical - self.height * dist_multiplier))
        elif direction == "down":
            end_x_logical = x_logical
            end_y_logical = min(self.height, int(y_logical + self.height * dist_multiplier))
        elif direction == "left":
            end_x_logical = max(0, int(x_logical - self.width * dist_multiplier))
            end_y_logical = y_logical
        elif direction == "right":
            end_x_logical = min(self.width, int(x_logical + self.width * dist_multiplier))
            end_y_logical = y_logical
        else:
            end_x_logical = x_logical
            end_y_logical = min(self.height, int(y_logical + self.height * dist_multiplier))

        end_x, end_y = int(end_x_logical * SCALE_FACTOR), int(end_y_logical * SCALE_FACTOR)
        return self.action_handler.swipe(x, y, end_x, end_y)

    def back(self) -> bool:
        """Navigate back (swipe from left edge on iOS)."""
        return self.action_handler.back()

    def home(self) -> bool:
        """Press home button."""
        return self.action_handler.home()

    def enter(self) -> bool:
        """Press Enter key (hides keyboard on iOS)."""
        return self.action_handler.hide_keyboard()

    def launch_app(self, app_name: str) -> bool:
        """Launch an app by name."""
        return self.action_handler.launch_app(app_name)

    def save_screenshot(self, file_path: str) -> bool:
        """Save screenshot to file."""
        screenshot = get_screenshot(wda_url=self.wda_url, session_id=self.session_id)
        return save_screenshot(screenshot, file_path)

    def get_screenshot(self) -> Screenshot:
        """Get current screenshot."""
        return get_screenshot(wda_url=self.wda_url, session_id=self.session_id)

    def get_xml(self, prefix: str = "", save_dir: str = "") -> str:
        """
        Get iOS page source (XML hierarchy).

        Compatible with Android-Lab's get_xml interface.
        """
        try:
            xml_string = get_page_source(
                wda_url=self.wda_url,
                session_id=self.session_id,
                timeout=15
            )

            if xml_string:
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    xml_path = os.path.join(save_dir, f"{prefix}.xml")
                    try:
                        with open(xml_path, 'w', encoding='utf-8') as f:
                            f.write(xml_string)
                    except Exception as e:
                        print(f"Warning: Failed to save XML to {xml_path}: {e}")

                return "SUCCESS"
            else:
                return "ERROR: Failed to get page source (returned None)"
        except Exception as e:
            print(f"Error getting XML: {e}")
            return f"ERROR: {e}"
