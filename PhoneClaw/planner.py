"""Task Planner for PhoneClaw.

Uses an LLM to decompose a high-level task description into an ordered list of
atomic subtasks, each with a visually-verifiable success criterion.
"""

import json
import re
from typing import List, Optional

from PhoneClaw.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_TEMPLATE
from PhoneClaw.state import SubTask


class TaskPlanner:
    """
    Calls an LLM to break a task into subtasks with success criteria.

    The agent object must implement:
        agent.act(messages: list[dict]) -> str
    where messages follow the OpenAI chat format.
    """

    def __init__(self, agent, max_retries: int = 3):
        """
        Args:
            agent: LLM agent instance (e.g., OpenAIAgent or QwenVLAgent from Android-Lab).
            max_retries: How many times to retry if JSON parsing fails.
        """
        self.agent = agent
        self.max_retries = max_retries

    def plan(self, task: str, user_context: str = "") -> List[SubTask]:
        """
        Decompose a task into ordered subtasks.

        Args:
            task: High-level task description.
            user_context: Optional background about the user (from UserMemory).
                          Injected into the system prompt so the planner can make
                          more informed decisions (preferred apps, location, etc.).

        Returns:
            List of SubTask objects, ordered from first to last.

        Raises:
            ValueError: If the LLM fails to return a valid subtask list after all retries.
        """
        user_content = PLANNER_USER_TEMPLATE.format(task=task)

        # Render user context section; falls back to empty string if nothing known yet
        context_block = (
            user_context.strip() + "\n\n"
            if user_context and user_context.strip()
            else ""
        )
        # Use replace() instead of .format() to avoid KeyError on the JSON
        # example curly-braces inside PLANNER_SYSTEM_PROMPT
        system_content = PLANNER_SYSTEM_PROMPT.replace("{user_context}", context_block)

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.agent.act(messages)
                subtasks = self._parse_response(response)
                if subtasks:
                    print(f"[Planner] Decomposed task into {len(subtasks)} subtask(s).")
                    for st in subtasks:
                        print(f"  #{st.id}: {st.instruction}")
                    return subtasks
                else:
                    last_error = f"Attempt {attempt}: Parsed 0 subtasks from response."
                    print(f"[Planner] Warning: {last_error}")
            except Exception as e:
                last_error = f"Attempt {attempt}: {e}"
                print(f"[Planner] Error during planning: {last_error}")

        # Final fallback: treat the entire task as a single subtask
        print(f"[Planner] All {self.max_retries} attempts failed. Falling back to single subtask.")
        fallback = SubTask(
            id=1,
            instruction=task,
            success_criteria="The task appears to be completed as described."
        )
        return [fallback]

    def _parse_response(self, response: str) -> List[SubTask]:
        """
        Parse LLM response into SubTask objects.

        Handles:
        - Clean JSON array
        - JSON wrapped in markdown code fences
        - JSON embedded in prose
        """
        text = response.strip()

        # Strip markdown code fences if present
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        # Try to extract a JSON array even if surrounded by prose
        array_match = re.search(r'\[.*\]', text, re.DOTALL)
        if array_match:
            text = array_match.group(0)

        data = json.loads(text)

        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")

        subtasks = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not a dict: {item}")

            subtask_id = item.get("id", i + 1)
            instruction = item.get("instruction", "").strip()
            success_criteria = item.get("success_criteria", "").strip()

            if not instruction:
                raise ValueError(f"Item {i} has no 'instruction' field")
            if not success_criteria:
                raise ValueError(f"Item {i} has no 'success_criteria' field")

            subtasks.append(SubTask(
                id=subtask_id,
                instruction=instruction,
                success_criteria=success_criteria,
            ))

        return subtasks
