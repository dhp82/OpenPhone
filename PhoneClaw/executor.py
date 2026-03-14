"""iOS executor for PhoneClaw - adapts iOS actions to Android-Lab interface."""

import os
import time
from typing import Optional, List

from PhoneClaw.actions import IOSActionHandler, _physical_to_logical, _logical_to_physical
from PhoneClaw.screenshot import get_screenshot, save_screenshot, Screenshot
from PhoneClaw.hierarchy import IOSElement, get_page_source, get_ios_elements


class IOSExecutor:
    """
    iOS executor that adapts iOS device control to Android-Lab's executor interface.
    """

    def __init__(self, wda_url: str = "http://localhost:8100", session_id: Optional[str] = None):
        self.action_handler = IOSActionHandler(wda_url=wda_url, session_id=session_id)
        self.wda_url = wda_url
        self.session_id = session_id
        self.current_screenshot: Optional[Screenshot] = None
        self.current_return = None
        self.is_finish = False
        self.finish_message: Optional[str] = None
        self.elem_list: List[IOSElement] = []
        self.current_screenshot_path: Optional[str] = None
        # Cached physical screen size (set lazily by _get_screen_physical_size)
        self._phys_w: Optional[int] = None
        self._phys_h: Optional[int] = None

    def get_screenshot(self) -> Screenshot:
        """Get current screenshot."""
        self.current_screenshot = get_screenshot(
            wda_url=self.wda_url,
            session_id=self.session_id,
        )
        return self.current_screenshot

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _get_screen_physical_size(self) -> tuple[int, int]:
        """
        Return the physical (pixel) screen dimensions, queried once and cached.

        WDA reports logical coordinates; physical = logical × SCALE_FACTOR.
        Uses the actual screenshot file when available for maximum accuracy
        (handles non-standard scale factors such as iPhone SE's 2×).
        Falls back to WDA logical × SCALE_FACTOR when no screenshot is saved yet.
        """
        if self._phys_w and self._phys_h:
            return self._phys_w, self._phys_h

        # Try reading dimensions from the most recent screenshot file
        if self.current_screenshot_path and os.path.exists(self.current_screenshot_path):
            try:
                import cv2
                img = cv2.imread(self.current_screenshot_path)
                if img is not None:
                    h, w = img.shape[:2]
                    self._phys_w, self._phys_h = w, h
                    return self._phys_w, self._phys_h
            except Exception:
                pass

        # Fallback: WDA logical size × SCALE_FACTOR
        from PhoneClaw.actions import SCALE_FACTOR
        w_logical, h_logical = self.action_handler.get_screen_size()
        self._phys_w = w_logical * SCALE_FACTOR
        self._phys_h = h_logical * SCALE_FACTOR
        return self._phys_w, self._phys_h

    def _rel_to_physical(self, rx: float, ry: float) -> tuple[int, int]:
        """
        Convert normalized relative coordinates [0, 1] to physical pixel coordinates.

        (0.0, 0.0) → top-left corner
        (1.0, 1.0) → bottom-right corner
        Values are clamped to [0, 1] before conversion.
        """
        rx = max(0.0, min(1.0, float(rx)))
        ry = max(0.0, min(1.0, float(ry)))
        w, h = self._get_screen_physical_size()
        return round(rx * w), round(ry * h)

    def tap(self, x: int, y: int) -> dict:
        """Tap at coordinates (x, y)."""
        success = self.action_handler.tap(x, y)
        self.current_return = {
            "operation": "do",
            "action": "Tap",
            "kwargs": {"element": [x, y]}
        }
        return self.current_return

    def text(self, input_str: str) -> dict:
        """Type text into the currently focused input field."""
        self.action_handler.clear_text()
        time.sleep(0.5)
        success = self.action_handler.type_text(input_str)
        time.sleep(0.5)
        self.action_handler.hide_keyboard()
        time.sleep(0.5)
        self.current_return = {
            "operation": "do",
            "action": "Type",
            "kwargs": {"text": input_str}
        }
        return self.current_return

    def type(self, input_str: str) -> dict:
        """Alias for text method."""
        return self.text(input_str)

    def long_press(self, x: int, y: int) -> dict:
        """Long press at coordinates (x, y)."""
        success = self.action_handler.long_press(x, y)
        self.current_return = {
            "operation": "do",
            "action": "Long Press",
            "kwargs": {"element": [x, y]}
        }
        return self.current_return

    def swipe(self, x: int, y: int, direction: str, dist: str = "medium") -> dict:
        """Swipe from coordinates (x, y) in a named direction (legacy / do() interface)."""
        screen_width_logical, screen_height_logical = self.action_handler.get_screen_size()

        x_logical, y_logical = _physical_to_logical(x, y)

        dist_multiplier = {"short": 0.3, "medium": 0.5, "long": 0.7}.get(dist, 0.5)

        if direction == "up":
            end_x_logical = x_logical
            end_y_logical = max(0, int(y_logical - screen_height_logical * dist_multiplier))
        elif direction == "down":
            end_x_logical = x_logical
            end_y_logical = min(screen_height_logical, int(y_logical + screen_height_logical * dist_multiplier))
        elif direction == "left":
            end_x_logical = max(0, int(x_logical - screen_width_logical * dist_multiplier))
            end_y_logical = y_logical
        elif direction == "right":
            end_x_logical = min(screen_width_logical, int(x_logical + screen_width_logical * dist_multiplier))
            end_y_logical = y_logical
        else:
            end_x_logical = x_logical
            end_y_logical = min(screen_height_logical, int(y_logical + screen_height_logical * dist_multiplier))

        end_x, end_y = _logical_to_physical(end_x_logical, end_y_logical)

        success = self.action_handler.swipe(x, y, end_x, end_y)
        self.current_return = {
            "operation": "do",
            "action": "Swipe",
            "kwargs": {
                "element": [x, y],
                "direction": direction,
                "dist": dist
            }
        }
        return self.current_return

    def swipe_coords(self, x1: int, y1: int, x2: int, y2: int) -> dict:
        """
        Swipe from (x1, y1) to (x2, y2) using explicit physical pixel coordinates.

        This is the primary swipe method used by the coordinate-based agent.
        Both points are in screenshot physical coordinates; conversion to WDA
        logical coordinates is handled internally by action_handler.swipe().
        """
        success = self.action_handler.swipe(x1, y1, x2, y2)
        self.current_return = {
            "operation": "do",
            "action": "Swipe",
            "kwargs": {"from": [x1, y1], "to": [x2, y2]}
        }
        return self.current_return

    def back(self) -> dict:
        """Navigate back (swipe from left edge on iOS)."""
        success = self.action_handler.back()
        self.current_return = {
            "operation": "do",
            "action": "Back",
            "kwargs": {}
        }
        return self.current_return

    def home(self) -> dict:
        """Press the home button."""
        success = self.action_handler.home()
        self.current_return = {
            "operation": "do",
            "action": "Home",
            "kwargs": {}
        }
        return self.current_return

    def wait(self, interval: int = 5) -> dict:
        """Wait for specified interval."""
        if interval < 0 or interval > 10:
            interval = 5
        time.sleep(interval)
        self.current_return = {
            "operation": "do",
            "action": "Wait",
            "kwargs": {"interval": interval}
        }
        return self.current_return

    def enter(self) -> dict:
        """Press Enter key (hides keyboard on iOS)."""
        self.action_handler.hide_keyboard()
        self.current_return = {
            "operation": "do",
            "action": "Enter",
            "kwargs": {}
        }
        return self.current_return

    def launch(self, app_name: str) -> dict:
        """Launch an app by name."""
        success = self.action_handler.launch_app(app_name)
        self.current_return = {
            "operation": "do",
            "action": "Launch",
            "kwargs": {"app_name": app_name}
        }
        return self.current_return

    def finish(self, message: Optional[str] = None) -> dict:
        """Finish the current subtask (used by executor code snippets)."""
        self.is_finish = True
        if message:
            self.finish_message = message
        self.current_return = {
            "operation": "finish",
            "action": "finish",
            "kwargs": {"message": message}
        }
        return self.current_return

    def reset_finish(self):
        """Reset is_finish flag for next subtask."""
        self.is_finish = False
        self.finish_message = None

    def get_current_app(self) -> str:
        """Get the currently active app name."""
        return self.action_handler.get_current_app()

    def get_screen_size(self) -> tuple[int, int]:
        """Get the screen dimensions."""
        return self.action_handler.get_screen_size()

    def set_elem_list(self, xml_path_or_string: str):
        """Set element list from iOS XML source."""
        if os.path.exists(xml_path_or_string):
            with open(xml_path_or_string, 'r', encoding='utf-8') as f:
                xml_string = f.read()
        else:
            xml_string = xml_path_or_string

        self.elem_list = get_ios_elements(xml_string)

    def tap_by_index(self, index: int) -> dict:
        """Tap element by index (1-based)."""
        if not self.elem_list:
            error_msg = (
                "Element list is empty. Please ensure XML is parsed and set_elem_list() is called."
            )
            print(f"Error: {error_msg}")
            self.current_return = {
                "operation": "error",
                "action": "Tap",
                "kwargs": {"index": index, "error": error_msg}
            }
            raise ValueError(error_msg)
        assert 0 < index <= len(self.elem_list), f"Tap Index {index} out of range (available: 1-{len(self.elem_list)})"

        tl, br = self.elem_list[index - 1].bbox
        x_logical, y_logical = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
        x, y = _logical_to_physical(x_logical, y_logical)

        return self.tap(x, y)

    def long_press_by_index(self, index: int) -> dict:
        """Long press element by index (1-based)."""
        if not self.elem_list:
            raise ValueError("Element list is empty. Please ensure XML is parsed and set_elem_list() is called.")
        assert 0 < index <= len(self.elem_list), f"Long Press Index {index} out of range (available: 1-{len(self.elem_list)})"

        tl, br = self.elem_list[index - 1].bbox
        x_logical, y_logical = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
        x, y = _logical_to_physical(x_logical, y_logical)

        return self.long_press(x, y)

    def swipe_by_index(self, index: int, direction: str, dist: str = "medium") -> dict:
        """Swipe element by index (1-based)."""
        if not self.elem_list:
            raise ValueError("Element list is empty. Please ensure XML is parsed and set_elem_list() is called.")
        assert 0 < index <= len(self.elem_list), f"Swipe Index {index} out of range (available: 1-{len(self.elem_list)})"

        tl, br = self.elem_list[index - 1].bbox
        x_logical, y_logical = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
        x, y = _logical_to_physical(x_logical, y_logical)

        return self.swipe(x, y, direction, dist)

    def __call__(self, code_snippet: str):
        """
        Execute a coordinate-based code snippet from the VLM.

        The VLM outputs normalized relative coordinates in [0, 1].
        This method converts them to physical pixels before dispatching.

        Supported calls:
            tap(rx, ry)                      - tap at relative position (rx, ry)
            long_press(rx, ry)               - long press at relative position
            swipe(rx1, ry1, rx2, ry2)        - swipe from rel (rx1,ry1) to (rx2,ry2)
            type("text")  /  text("text")    - type text
            back()
            home()
            wait(seconds)
            finish("message")

        All rx/ry values are floats in [0, 1]:
            (0.0, 0.0) = top-left corner
            (1.0, 1.0) = bottom-right corner
        """
        import re

        if not code_snippet:
            print("Warning: code_snippet is empty or None, skipping execution")
            self.current_return = {
                "operation": "skip",
                "action": "skip",
                "kwargs": {"reason": "Empty code snippet"}
            }
            return self.current_return

        # --- Relative-coordinate wrappers ---
        # These accept [0,1] floats from the VLM and convert to physical pixels
        # before forwarding to the underlying executor methods.

        def _tap(rx, ry):
            px, py = self._rel_to_physical(rx, ry)
            print(f"[Exec] tap(rel=({rx:.3f},{ry:.3f}) → phys=({px},{py}))")
            return self.tap(px, py)

        def _long_press(rx, ry):
            px, py = self._rel_to_physical(rx, ry)
            print(f"[Exec] long_press(rel=({rx:.3f},{ry:.3f}) → phys=({px},{py}))")
            return self.long_press(px, py)

        def _swipe(rx1, ry1, rx2, ry2):
            px1, py1 = self._rel_to_physical(rx1, ry1)
            px2, py2 = self._rel_to_physical(rx2, ry2)
            print(f"[Exec] swipe(rel=({rx1:.3f},{ry1:.3f})→({rx2:.3f},{ry2:.3f})"
                  f" → phys=({px1},{py1})→({px2},{py2}))")
            return self.swipe_coords(px1, py1, px2, py2)

        local_context = {
            'tap':        _tap,
            'long_press': _long_press,
            'swipe':      _swipe,
            'type':       self.text,
            'text':       self.text,
            'back':       self.back,
            'home':       self.home,
            'wait':       self.wait,
            'finish':     self.finish,
            'launch':     self.launch,
        }

        # Strip accidental leading zeros from integer literals (e.g. 01 → 1) that
        # would be Python SyntaxErrors.
        # IMPORTANT: use a negative lookbehind (?<!\.) so that decimal fractions
        # such as 0.095 or 0.06 are NOT touched.  Without it the word boundary
        # between the decimal point and the digit would cause:
        #   0.095 → 0.95   (0.095 interpreted as "09" with leading zero stripped)
        #   0.06  → 0.6    (same issue)
        code_snippet = re.sub(r'(?<!\.)\b0+(\d)', r'\1', code_snippet)

        try:
            exec(code_snippet, {}, local_context)
        except Exception as e:
            print(f"Error executing code snippet '{code_snippet}': {e}")
            import traceback
            traceback.print_exc()
            self.current_return = {
                "operation": "error",
                "action": "error",
                "kwargs": {"error": str(e), "code": code_snippet}
            }

        return self.current_return

    def do(self, action=None, element=None, **kwargs):
        """Execute an action - compatible with Android-Lab's do() interface."""
        assert action in [
            "Tap", "Type", "Swipe", "Enter", "Home", "Back", "Long Press", "Wait", "Launch", "Call_API"
        ], f"Unsupported Action: {action}"

        if action == "Tap":
            if isinstance(element, list) and len(element) == 4:
                center_x = (element[0] + element[2]) / 2
                center_y = (element[1] + element[3]) / 2
            elif isinstance(element, list) and len(element) == 2:
                center_x, center_y = element
            else:
                raise ValueError("Invalid element format for Tap")
            return self.tap(int(center_x), int(center_y))

        elif action == "Type":
            assert "text" in kwargs, "text is required for Type action"
            return self.text(kwargs["text"])

        elif action == "Swipe":
            assert "direction" in kwargs, "direction is required for Swipe action"
            if element is None:
                screen_width_logical, screen_height_logical = self.get_screen_size()
                center_x, center_y = _logical_to_physical(
                    screen_width_logical // 2,
                    screen_height_logical // 2
                )
            elif isinstance(element, list) and len(element) == 4:
                center_x = (element[0] + element[2]) / 2
                center_y = (element[1] + element[3]) / 2
            elif isinstance(element, list) and len(element) == 2:
                center_x, center_y = element
            else:
                raise ValueError("Invalid element format for Swipe")
            dist = kwargs.get("dist", "medium")
            return self.swipe(int(center_x), int(center_y), kwargs["direction"], dist)

        elif action == "Enter":
            return self.enter()

        elif action == "Home":
            return self.home()

        elif action == "Back":
            return self.back()

        elif action == "Long Press":
            if isinstance(element, list) and len(element) == 4:
                center_x = (element[0] + element[2]) / 2
                center_y = (element[1] + element[3]) / 2
            elif isinstance(element, list) and len(element) == 2:
                center_x, center_y = element
            else:
                raise ValueError("Invalid element format for Long Press")
            return self.long_press(int(center_x), int(center_y))

        elif action == "Wait":
            interval = kwargs.get("interval", 5)
            return self.wait(interval)

        elif action == "Launch":
            assert "app" in kwargs or "app_name" in kwargs, "app or app_name is required for Launch action"
            app_name = kwargs.get("app") or kwargs.get("app_name")
            return self.launch(app_name)

        elif action == "Call_API":
            instruction = kwargs.get("instruction", "")
            with_screen_info = kwargs.get("with_screen_info", True)
            self.current_return = {
                "operation": "do",
                "action": "Call_API",
                "kwargs": {
                    "instruction": instruction,
                    "with_screen_info": with_screen_info
                }
            }
            return self.current_return

        else:
            raise NotImplementedError(f"Action {action} not implemented")

    def update_screenshot(self, prefix=None, suffix=None):
        """Update screenshot and save to screenshot_dir."""
        screenshot = self.get_screenshot()

        if hasattr(self, 'screenshot_dir'):
            timestamp = time.time()
            if prefix is None and suffix is None:
                screenshot_path = f"{self.screenshot_dir}/screenshot-{timestamp}.png"
            elif prefix is not None and suffix is None:
                screenshot_path = f"{self.screenshot_dir}/screenshot-{prefix}-{timestamp}.png"
            elif prefix is None and suffix is not None:
                screenshot_path = f"{self.screenshot_dir}/screenshot-{timestamp}-{suffix}.png"
            else:
                screenshot_path = f"{self.screenshot_dir}/screenshot-{prefix}-{timestamp}-{suffix}.png"

            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            save_screenshot(screenshot, screenshot_path)
            self.current_screenshot_path = screenshot_path
            self.current_screenshot = screenshot_path

        return screenshot
