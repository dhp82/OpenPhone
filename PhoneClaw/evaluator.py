"""Evaluator for PhoneClaw Ralph Loop.

Uses a VLM to determine whether the current screen satisfies a subtask's
success criteria.  Returns a structured pass/fail result with a reason.
"""

import json
import re
from typing import Optional, Dict, Any

from PhoneClaw.prompts import EVALUATOR_SYSTEM_PROMPT, EVALUATOR_USER_TEMPLATE


class EvalResult:
    """Result from the Evaluator."""

    def __init__(self, passed: bool, reason: str):
        self.passed = passed
        self.reason = reason

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "reason": self.reason}

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"EvalResult({status}: {self.reason})"


class SubTaskEvaluator:
    """
    Evaluates whether a screenshot satisfies a subtask's success criterion.

    The agent object must implement:
        agent.prompt_to_message_visual(prompt: str, image_path: str) -> list[dict]
        agent.act(messages: list[dict]) -> str
    """

    def __init__(self, agent, max_retries: int = 2):
        """
        Args:
            agent: VLM agent instance (e.g., OpenAIAgent from Android-Lab).
            max_retries: How many times to retry if JSON parsing fails.
        """
        self.agent = agent
        self.max_retries = max_retries

    def evaluate(
        self,
        screenshot_path: str,
        success_criteria: str,
    ) -> EvalResult:
        """
        Evaluate whether the screenshot satisfies the success criterion.

        Args:
            screenshot_path: Path to the current screenshot (labeled or plain).
            success_criteria: The success criterion text for the current subtask.

        Returns:
            EvalResult with passed (bool) and reason (str).
        """
        if not screenshot_path:
            return EvalResult(passed=False, reason="No screenshot available for evaluation.")

        user_prompt = EVALUATOR_USER_TEMPLATE.format(success_criteria=success_criteria)

        system_msg = {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT}

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                user_messages = self.agent.prompt_to_message_visual(user_prompt, screenshot_path)
                messages = [system_msg, *user_messages]
                response = self.agent.act(messages)

                result = self._parse_response(response)
                return result

            except Exception as e:
                last_error = str(e)
                print(f"[Evaluator] Attempt {attempt} failed: {e}")

        # Fallback: conservative fail
        print(f"[Evaluator] All {self.max_retries} attempts failed. Defaulting to FAIL.")
        return EvalResult(
            passed=False,
            reason=f"Evaluator failed to produce a valid response. Last error: {last_error}"
        )

    def _parse_response(self, response: str) -> EvalResult:
        """
        Parse the LLM response into an EvalResult.

        Handles:
        - Clean JSON object
        - JSON wrapped in markdown code fences
        - JSON embedded in prose
        """
        text = response.strip()

        # Strip markdown code fences
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        # Try to extract a JSON object even if surrounded by prose
        obj_match = re.search(r'\{.*\}', text, re.DOTALL)
        if obj_match:
            text = obj_match.group(0)

        data = json.loads(text)

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        passed = data.get("passed")
        reason = data.get("reason", "").strip()

        if passed is None:
            raise ValueError("JSON object missing 'passed' field")

        # Normalize: accept string "true"/"false" as well as booleans
        if isinstance(passed, str):
            passed = passed.lower() in ("true", "1", "yes")
        else:
            passed = bool(passed)

        if not reason:
            reason = "No reason provided."

        return EvalResult(passed=passed, reason=reason)
