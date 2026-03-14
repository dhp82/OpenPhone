"""iOS device connection management via WebDriverAgent."""

import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ConnectionType(Enum):
    """Type of iOS connection."""
    USB = "usb"
    NETWORK = "network"


@dataclass
class DeviceInfo:
    """Information about a connected iOS device."""
    device_id: str
    status: str
    connection_type: ConnectionType
    model: Optional[str] = None
    ios_version: Optional[str] = None
    device_name: Optional[str] = None


class IOSConnection:
    """
    Manages connections to iOS devices via libimobiledevice and WebDriverAgent.

    Requires:
        - libimobiledevice (idevice_id, ideviceinfo)
        - WebDriverAgent running on the iOS device
    """

    def __init__(self, wda_url: str = "http://localhost:8100"):
        self.wda_url = wda_url.rstrip("/")
        self.session_id: Optional[str] = None

    def list_devices(self) -> list[DeviceInfo]:
        """List all connected iOS devices."""
        try:
            result = subprocess.run(
                ["idevice_id", "-ln"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            devices = []
            for line in result.stdout.strip().split("\n"):
                udid = line.strip()
                if not udid:
                    continue

                conn_type = (
                    ConnectionType.NETWORK
                    if "-" in udid and len(udid) > 40
                    else ConnectionType.USB
                )

                device_info = self._get_device_details(udid)

                devices.append(
                    DeviceInfo(
                        device_id=udid,
                        status="connected",
                        connection_type=conn_type,
                        model=device_info.get("model"),
                        ios_version=device_info.get("ios_version"),
                        device_name=device_info.get("name"),
                    )
                )

            return devices

        except FileNotFoundError:
            print(
                "Error: idevice_id not found. Install libimobiledevice: brew install libimobiledevice"
            )
            return []
        except Exception as e:
            print(f"Error listing devices: {e}")
            return []

    def _get_device_details(self, udid: str) -> dict[str, str]:
        """Get detailed information about a specific device."""
        try:
            result = subprocess.run(
                ["ideviceinfo", "-u", udid],
                capture_output=True,
                text=True,
                timeout=5,
            )

            info = {}
            for line in result.stdout.split("\n"):
                if ": " in line:
                    key, value = line.split(": ", 1)
                    key = key.strip()
                    value = value.strip()

                    if key == "ProductType":
                        info["model"] = value
                    elif key == "ProductVersion":
                        info["ios_version"] = value
                    elif key == "DeviceName":
                        info["name"] = value

            return info

        except Exception:
            return {}

    def is_connected(self, device_id: Optional[str] = None) -> bool:
        """Check if a device is connected."""
        devices = self.list_devices()
        if not devices:
            return False
        if device_id is None:
            return len(devices) > 0
        return any(d.device_id == device_id for d in devices)

    def is_wda_ready(self, timeout: int = 2) -> bool:
        """Check if WebDriverAgent is running and accessible."""
        try:
            import requests
            response = requests.get(
                f"{self.wda_url}/status", timeout=timeout, verify=False
            )
            return response.status_code == 200
        except ImportError:
            print("Error: requests library not found. Install it: pip install requests")
            return False
        except Exception:
            return False

    def start_wda_session(self) -> tuple[bool, str]:
        """Start a new WebDriverAgent session."""
        try:
            import requests
            response = requests.post(
                f"{self.wda_url}/session",
                json={"capabilities": {}},
                timeout=30,
                verify=False,
            )

            if response.status_code in (200, 201):
                data = response.json()
                session_id = data.get("sessionId") or data.get("value", {}).get("sessionId")
                if session_id:
                    self.session_id = session_id
                return True, session_id or "session_started"
            else:
                return False, f"Failed to start session: {response.text}"

        except ImportError:
            return (
                False,
                "requests library not found. Install it: pip install requests",
            )
        except Exception as e:
            return False, f"Error starting WDA session: {e}"

    def get_wda_status(self) -> dict:
        """Get WebDriverAgent status information."""
        try:
            import requests
            response = requests.get(f"{self.wda_url}/status", timeout=5, verify=False)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception:
            return {}
