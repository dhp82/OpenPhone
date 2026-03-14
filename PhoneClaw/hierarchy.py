"""iOS UI Hierarchy - get and parse iOS page source for element labeling."""

import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class IOSElement:
    """Represents an iOS UI element."""
    uid: str
    bbox: Tuple[Tuple[int, int], Tuple[int, int]]  # ((x1, y1), (x2, y2))
    attrib: str  # "clickable" or "focusable"
    element_type: str  # XCUIElementTypeButton, etc.
    name: Optional[str] = None
    label: Optional[str] = None
    identifier: Optional[str] = None


def get_page_source(
    wda_url: str = "http://localhost:8100",
    session_id: Optional[str] = None,
    timeout: int = 10
) -> Optional[str]:
    """
    Get iOS page source (XML hierarchy) via WebDriverAgent.

    Args:
        wda_url: WebDriverAgent URL.
        session_id: Optional WDA session ID.
        timeout: Request timeout in seconds.

    Returns:
        XML string of the page source, or None if failed.
    """
    try:
        import requests

        urls_to_try = []
        if session_id:
            urls_to_try.append(f"{wda_url.rstrip('/')}/session/{session_id}/source")
        urls_to_try.append(f"{wda_url.rstrip('/')}/source")

        last_error = None
        for url in urls_to_try:
            try:
                response = requests.get(url, timeout=timeout, verify=False)

                if response.status_code == 200:
                    try:
                        data = response.json()
                        source = None

                        if isinstance(data, dict):
                            source = data.get("value")

                            if isinstance(source, dict):
                                source = source.get("source") or source.get("value")

                            if source is None:
                                source = data.get("source")

                            if source is None and isinstance(data.get("value"), dict):
                                source = data.get("value", {}).get("source")

                            if isinstance(source, dict):
                                source = source.get("source") or source.get("value")
                        else:
                            source = str(data) if data else None

                        if source and isinstance(source, str) and len(source.strip()) > 0:
                            source = source.strip()
                            if (source.startswith('"') and source.endswith('"')) or \
                               (source.startswith("'") and source.endswith("'")):
                                source = source[1:-1]

                            source = source.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
                            source = source.replace('\\"', '"').replace("\\'", "'")

                            source_stripped = source.strip()
                            if source_stripped.startswith('<') or '<?xml' in source_stripped[:100]:
                                try:
                                    ET.fromstring(source_stripped)
                                    return source
                                except ET.ParseError:
                                    print(f"Warning: XML from {url} may have parsing issues, returning anyway")
                                    return source
                            else:
                                return source
                        else:
                            print(f"Warning: Empty or invalid page source from {url}")

                    except ValueError:
                        if response.text and len(response.text.strip()) > 0:
                            text = response.text.strip()
                            if text.startswith('<') or '<?xml' in text[:100]:
                                return text

                elif response.status_code == 404:
                    continue
                elif response.status_code == 500:
                    last_error = f"Server error (500) from {url}"
                    continue
                else:
                    last_error = f"HTTP {response.status_code} from {url}"
                    continue

            except Exception as e:
                last_error = f"Error getting page source from {url}: {e}"
                continue

        if last_error:
            print(f"Failed to get page source. Last error: {last_error}")
        return None

    except ImportError:
        print("Error: requests library required. Install: pip install requests")
        return None
    except Exception as e:
        print(f"Error getting page source: {e}")
        return None


def parse_bounds(bounds_str: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Parse bounds string from iOS XML.

    iOS bounds format: "{{x, y}, {width, height}}" or "x,y,width,height"
    """
    if not bounds_str:
        return None

    try:
        if "{{" in bounds_str:
            bounds_str = bounds_str.replace("{{", "").replace("}}", "").replace("{", "").replace("}", "")
            parts = bounds_str.split(",")
            if len(parts) >= 4:
                x = int(float(parts[0].strip()))
                y = int(float(parts[1].strip()))
                width = int(float(parts[2].strip()))
                height = int(float(parts[3].strip()))
                return ((x, y), (x + width, y + height))

        parts = bounds_str.split(",")
        if len(parts) >= 4:
            x = int(float(parts[0].strip()))
            y = int(float(parts[1].strip()))
            width = int(float(parts[2].strip()))
            height = int(float(parts[3].strip()))
            return ((x, y), (x + width, y + height))

        return None
    except Exception as e:
        print(f"Error parsing bounds '{bounds_str}': {e}")
        return None


def get_element_bounds(element: ET.Element) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Get bounds from an iOS XML element."""
    bounds_str = element.get('bounds', '')
    if bounds_str:
        bbox = parse_bounds(bounds_str)
        if bbox:
            return bbox

    try:
        x_str = element.get('x', '')
        y_str = element.get('y', '')
        width_str = element.get('width', '')
        height_str = element.get('height', '')

        if x_str and y_str and width_str and height_str:
            x = int(float(x_str))
            y = int(float(y_str))
            width = int(float(width_str))
            height = int(float(height_str))
            return ((x, y), (x + width, y + height))
    except (ValueError, TypeError):
        pass

    return None


def get_element_id(element: ET.Element) -> str:
    """Generate a unique ID for an iOS element."""
    element_type = element.tag if hasattr(element, 'tag') else element.get('type', 'Unknown')

    bbox = get_element_bounds(element)
    if bbox:
        elem_w = bbox[1][0] - bbox[0][0]
        elem_h = bbox[1][1] - bbox[0][1]
    else:
        elem_w, elem_h = 0, 0

    identifier = element.get('name') or element.get('identifier') or element.get('label', '')

    if identifier:
        elem_id = f"{element_type}_{identifier.replace(' ', '_').replace(':', '_')}"
    else:
        elem_id = f"{element_type}_{elem_w}_{elem_h}"

    return elem_id


def is_interactive_element(element: ET.Element) -> bool:
    """Check if an iOS element is interactive (clickable/focusable)."""
    interactive_types = [
        'XCUIElementTypeButton',
        'XCUIElementTypeCell',
        'XCUIElementTypeTextField',
        'XCUIElementTypeSecureTextField',
        'XCUIElementTypeSearchField',
        'XCUIElementTypeSlider',
        'XCUIElementTypeSwitch',
        'XCUIElementTypeTab',
        'XCUIElementTypeLink',
        'XCUIElementTypeImage',
        'XCUIElementTypeIcon',
        'XCUIElementTypeStaticText',
    ]

    element_type = element.tag if hasattr(element, 'tag') else element.get('type', '')

    is_interactive_type = any(interactive_type in element_type for interactive_type in interactive_types)

    if not is_interactive_type:
        return False

    enabled = element.get('enabled', 'true')
    if enabled == 'false':
        return False

    visible = element.get('visible', 'true')
    if visible == 'false':
        return False

    bbox = get_element_bounds(element)
    if not bbox:
        return False

    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return False

    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return False

    return True


def traverse_ios_tree(
    xml_string: str,
    elem_list: List[IOSElement],
    attrib: str = "clickable",
    add_index: bool = False
):
    """Traverse iOS XML tree and extract interactive elements."""
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return
    except Exception:
        return

    def traverse(node, path=[]):
        path = path + [node]

        if is_interactive_element(node):
            bbox = get_element_bounds(node)

            if bbox:
                center = ((bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2)

                close = False
                for e in elem_list:
                    e_bbox = e.bbox
                    e_center = ((e_bbox[0][0] + e_bbox[1][0]) // 2,
                               (e_bbox[0][1] + e_bbox[1][1]) // 2)
                    dist = ((center[0] - e_center[0]) ** 2 +
                           (center[1] - e_center[1]) ** 2) ** 0.5
                    if dist <= 5:
                        close = True
                        break

                if not close:
                    elem_id = get_element_id(node)

                    if len(path) > 1:
                        parent_id = get_element_id(path[-2])
                        elem_id = f"{parent_id}_{elem_id}"

                    if add_index:
                        index = node.get('index', '0')
                        elem_id += f"_{index}"

                    element = IOSElement(
                        uid=elem_id,
                        bbox=bbox,
                        attrib=attrib,
                        element_type=node.tag if hasattr(node, 'tag') else node.get('type', ''),
                        name=node.get('name'),
                        label=node.get('label'),
                        identifier=node.get('identifier')
                    )
                    elem_list.append(element)

        for child in node:
            traverse(child, path)

    traverse(root)


def get_ios_elements(xml_string: str) -> List[IOSElement]:
    """Extract interactive elements from iOS XML."""
    if not xml_string or len(xml_string.strip()) == 0:
        return []

    clickable_list = []
    focusable_list = []

    traverse_ios_tree(xml_string, clickable_list, "clickable", True)
    traverse_ios_tree(xml_string, focusable_list, "focusable", True)

    elem_list = list(clickable_list)

    for elem in focusable_list:
        bbox = elem.bbox
        center = ((bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2)
        close = False
        for e in clickable_list:
            e_bbox = e.bbox
            e_center = ((e_bbox[0][0] + e_bbox[1][0]) // 2,
                       (e_bbox[0][1] + e_bbox[1][1]) // 2)
            dist = ((center[0] - e_center[0]) ** 2 +
                   (center[1] - e_center[1]) ** 2) ** 0.5
            if dist <= 10:
                close = True
                break
        if not close:
            elem_list.append(elem)

    return elem_list
