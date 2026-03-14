"""Recorder for PhoneClaw - logs per-step traces including Ralph Loop evaluation results."""

import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any

from PhoneClaw.screenshot import Screenshot
from PhoneClaw.labeling import draw_bbox_multi_ios


class PhoneClawRecorder:
    """
    Recorder that logs per-step traces for the PhoneClaw Ralph Loop agent.

    Extends ios_agent's IOSRecorder with Ralph Loop-specific fields:
      - subtask index and instruction
      - evaluator result (passed, reason)
      - fix attempt count
    """

    def __init__(self, task_id: str, instruction: str, page_executor, config=None):
        """
        Initialize recorder.

        Args:
            task_id: Unique task identifier.
            instruction: Top-level task instruction.
            page_executor: IOSExecutor instance.
            config: Optional config object with task_dir.
        """
        self.task_id = task_id
        self.instruction = instruction
        self.page_executor = page_executor

        self.turn_number = 0

        if config and hasattr(config, 'task_dir'):
            task_dir = config.task_dir
        else:
            task_dir = f"./phoneclaw_logs/{task_id}"

        trace_dir = os.path.join(task_dir, 'traces')
        screenshot_dir = os.path.join(task_dir, 'screenshots')
        xml_dir = os.path.join(task_dir, 'xml')

        os.makedirs(trace_dir, exist_ok=True)
        os.makedirs(screenshot_dir, exist_ok=True)
        os.makedirs(xml_dir, exist_ok=True)
        os.makedirs(task_dir, exist_ok=True)

        self.trace_file_path = os.path.join(trace_dir, 'trace.jsonl')
        self.screenshot_dir = screenshot_dir
        self.xml_dir = xml_dir
        self.log_dir = task_dir

        self.contents = []
        self.history = []
        self.current_screenshot_path: Optional[str] = None
        self.labeled_current_screenshot_path: Optional[str] = None
        self.xml_history = []

        # Ralph Loop specific tracking
        self.current_subtask_idx: int = 0
        self.current_subtask_instruction: str = ""
        self.current_subtask_criteria: str = ""

    def set_current_subtask(self, idx: int, instruction: str, criteria: str):
        """Update current subtask context for logging."""
        self.current_subtask_idx = idx
        self.current_subtask_instruction = instruction
        self.current_subtask_criteria = criteria

    def update_before(self, controller, need_screenshot: bool = False, need_labeled: bool = False, **kwargs):
        """
        Update recorder before action execution: capture XML, screenshot, and generate labeled image.

        Args:
            controller: IOSController instance.
            need_screenshot: Whether to capture screenshot.
            need_labeled: Whether to generate labeled screenshot.
        """
        xml_path = None
        xml_string = None

        xml_status = controller.get_xml(prefix=str(self.turn_number), save_dir=self.xml_dir)
        if "ERROR" not in xml_status and xml_status == "SUCCESS":
            xml_path = os.path.join(self.xml_dir, f"{self.turn_number}.xml")
            self.xml_history.append(xml_path)

        if need_screenshot:
            self.page_executor.update_screenshot(prefix=str(self.turn_number), suffix="before")
            self.current_screenshot_path = self.page_executor.current_screenshot_path

        # Element list parsing is only needed for legacy index-based execution.
        # The coordinate-based agent operates on raw screenshots directly, so we
        # skip this step to reduce latency.  Set to empty list for safety.
        self.page_executor.elem_list = []

        if need_labeled and self.current_screenshot_path:
            try:
                if not self.page_executor.elem_list:
                    self.labeled_current_screenshot_path = self.current_screenshot_path
                else:
                    labeled_path = self.current_screenshot_path.replace(".png", "_labeled.png")

                    import cv2
                    img = cv2.imread(self.current_screenshot_path)
                    scale_factor = None
                    if img is not None:
                        height, width = img.shape[:2]
                        if width >= 1100:
                            for logical_width in [375, 390, 393]:
                                if abs(width / logical_width - 3.0) < 0.1:
                                    scale_factor = width / logical_width
                                    break
                            if scale_factor is None:
                                scale_factor = width / 375.0
                        else:
                            scale_factor = 1.0

                    result = draw_bbox_multi_ios(
                        self.current_screenshot_path,
                        labeled_path,
                        self.page_executor.elem_list,
                        record_mode=False,
                        dark_mode=False,
                        scale_factor=scale_factor
                    )

                    if result is not None:
                        self.labeled_current_screenshot_path = labeled_path
                    else:
                        self.labeled_current_screenshot_path = self.current_screenshot_path

            except Exception:
                self.labeled_current_screenshot_path = self.current_screenshot_path
        elif need_labeled:
            self.labeled_current_screenshot_path = None

        step = {
            "trace_id": self.task_id,
            "index": self.turn_number,
            "subtask_idx": self.current_subtask_idx,
            "subtask_instruction": self.current_subtask_instruction,
            "subtask_criteria": self.current_subtask_criteria,
            "prompt": "** screenshot **" if self.turn_number > 0 else f"{self.instruction}",
            "image": self.current_screenshot_path,
            "labeled_image": self.labeled_current_screenshot_path if need_labeled else None,
            "xml": xml_path,
            "current_app": controller.get_current_app(),
            "window": controller.viewport_size,
            "target": self.instruction,
        }

        self.contents.append(step)

    def update_after_cot(
        self,
        exe_res,
        response: str,
        prompt_his: Optional[str] = None,
        code_snippet: Optional[str] = None,
        eval_result: Optional[Dict[str, Any]] = None,
        fix_attempt: int = 0,
    ):
        """
        Update recorder after action execution.

        Args:
            exe_res: Execution result from executor.
            response: Agent response.
            prompt_his: Prompt history from state assessment.
            code_snippet: Code snippet extracted from response.
            eval_result: Evaluator result dict with 'passed' and 'reason'.
            fix_attempt: Current fix attempt count for this subtask.
        """
        if self.contents:
            self.contents[-1]["response"] = response
            self.contents[-1]["execution_result"] = exe_res
            if prompt_his:
                self.contents[-1]["prompt_his"] = prompt_his
            if code_snippet:
                self.contents[-1]["code_snippet"] = code_snippet
            if eval_result is not None:
                self.contents[-1]["eval_result"] = eval_result
            self.contents[-1]["fix_attempt"] = fix_attempt

        if prompt_his:
            self.history.append(prompt_his)

        self._save_trace()

    def log_subtask_result(self, subtask_idx: int, subtask: dict, passed: bool, reason: str):
        """Log a subtask evaluation result as a separate JSONL entry."""
        entry = {
            "type": "subtask_result",
            "task_id": self.task_id,
            "subtask_idx": subtask_idx,
            "subtask_instruction": subtask.get("instruction", ""),
            "success_criteria": subtask.get("success_criteria", ""),
            "passed": passed,
            "reason": reason,
            "timestamp": time.time(),
        }
        with open(self.trace_file_path, 'a', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    def log_task_complete(self, all_passed: bool, summary: str, final_answer: str = None):
        """Log final task completion status and optional answer."""
        entry = {
            "type": "task_complete",
            "task_id": self.task_id,
            "instruction": self.instruction,
            "all_passed": all_passed,
            "summary": summary,
            "total_turns": self.turn_number,
            "timestamp": time.time(),
        }
        if final_answer:
            entry["final_answer"] = final_answer
        with open(self.trace_file_path, 'a', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    def get_latest_xml(self) -> str:
        """Get latest XML string from page source."""
        if self.xml_history:
            latest_xml_path = self.xml_history[-1]
            if os.path.exists(latest_xml_path):
                try:
                    with open(latest_xml_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    print(f"Error reading XML: {e}")
        return ""

    def _save_trace(self):
        """Save the latest trace step to JSONL file."""
        if self.contents:
            with open(self.trace_file_path, 'a', encoding='utf-8') as f:
                json.dump(self.contents[-1], f, ensure_ascii=False)
                f.write('\n')
