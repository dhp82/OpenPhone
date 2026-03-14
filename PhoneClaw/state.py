"""State persistence for PhoneClaw Ralph Loop.

Saves and restores task progress to the filesystem so that:
- Tasks can resume after interruption or context window exhaustion
- Each Ralph Loop iteration has access to full task history
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class SubTask:
    """A single atomic subtask with its success criterion and execution state."""
    id: int
    instruction: str
    success_criteria: str
    status: str = "pending"        # "pending" | "passed" | "failed" | "skipped"
    fix_retries: int = 0
    eval_reason: Optional[str] = None
    completed_at: Optional[float] = None


@dataclass
class TaskState:
    """Full state of a Ralph Loop task run."""
    task_id: str
    task_instruction: str
    subtasks: List[SubTask] = field(default_factory=list)
    current_subtask_idx: int = 0   # 0-based index into subtasks
    round_count: int = 0
    status: str = "running"        # "running" | "completed" | "failed"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ----- convenience helpers -----

    @property
    def current_subtask(self) -> Optional[SubTask]:
        """Return the subtask currently being worked on, or None if all done."""
        if self.current_subtask_idx < len(self.subtasks):
            return self.subtasks[self.current_subtask_idx]
        return None

    @property
    def is_complete(self) -> bool:
        """True when all subtasks are in a terminal state."""
        return self.current_subtask_idx >= len(self.subtasks)

    def advance(self):
        """Move to the next subtask."""
        self.current_subtask_idx += 1
        self.updated_at = time.time()

    def mark_current_passed(self, reason: str):
        """Mark the current subtask as passed."""
        st = self.current_subtask
        if st:
            st.status = "passed"
            st.eval_reason = reason
            st.completed_at = time.time()
        self.updated_at = time.time()

    def mark_current_failed(self, reason: str):
        """Mark the current subtask as failed (max retries exceeded)."""
        st = self.current_subtask
        if st:
            st.status = "failed"
            st.eval_reason = reason
            st.completed_at = time.time()
        self.updated_at = time.time()

    def increment_fix_retries(self):
        """Increment fix attempt counter for the current subtask."""
        st = self.current_subtask
        if st:
            st.fix_retries += 1
        self.updated_at = time.time()

    def summary(self) -> str:
        """Return a human-readable summary of task progress."""
        total = len(self.subtasks)
        passed = sum(1 for s in self.subtasks if s.status == "passed")
        failed = sum(1 for s in self.subtasks if s.status == "failed")
        pending = sum(1 for s in self.subtasks if s.status == "pending")
        lines = [
            f"Task: {self.task_instruction}",
            f"Progress: {passed}/{total} passed, {failed} failed, {pending} pending",
            f"Total rounds: {self.round_count}",
            f"Status: {self.status}",
        ]
        for i, st in enumerate(self.subtasks):
            marker = {
                "passed": "[PASS]",
                "failed": "[FAIL]",
                "pending": "[    ]",
                "skipped": "[SKIP]",
            }.get(st.status, "[    ]")
            reason_snippet = f" — {st.eval_reason[:60]}..." if st.eval_reason else ""
            lines.append(f"  {marker} #{st.id}: {st.instruction}{reason_snippet}")
        return "\n".join(lines)


class StateManager:
    """
    Manages task state persistence to the filesystem.

    State is stored as a JSON file at <state_dir>/phoneclaw_state.json.
    """

    STATE_FILENAME = "phoneclaw_state.json"

    def __init__(self, state_dir: str):
        """
        Args:
            state_dir: Directory where the state file will be stored (typically the task log dir).
        """
        self.state_dir = state_dir
        self.state_path = os.path.join(state_dir, self.STATE_FILENAME)
        os.makedirs(state_dir, exist_ok=True)

    # ----- serialization helpers -----

    def _subtask_to_dict(self, st: SubTask) -> Dict[str, Any]:
        return asdict(st)

    def _subtask_from_dict(self, d: Dict[str, Any]) -> SubTask:
        return SubTask(**d)

    def _state_to_dict(self, state: TaskState) -> Dict[str, Any]:
        d = {
            "task_id": state.task_id,
            "task_instruction": state.task_instruction,
            "subtasks": [self._subtask_to_dict(s) for s in state.subtasks],
            "current_subtask_idx": state.current_subtask_idx,
            "round_count": state.round_count,
            "status": state.status,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        }
        return d

    def _state_from_dict(self, d: Dict[str, Any]) -> TaskState:
        subtasks = [self._subtask_from_dict(s) for s in d.get("subtasks", [])]
        return TaskState(
            task_id=d["task_id"],
            task_instruction=d["task_instruction"],
            subtasks=subtasks,
            current_subtask_idx=d.get("current_subtask_idx", 0),
            round_count=d.get("round_count", 0),
            status=d.get("status", "running"),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
        )

    # ----- public API -----

    def save(self, state: TaskState):
        """Persist state to disk."""
        state.updated_at = time.time()
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(self._state_to_dict(state), f, ensure_ascii=False, indent=2)

    def load(self) -> Optional[TaskState]:
        """Load state from disk. Returns None if no state file exists."""
        if not os.path.exists(self.state_path):
            return None
        try:
            with open(self.state_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            return self._state_from_dict(d)
        except Exception as e:
            print(f"Warning: Failed to load state from {self.state_path}: {e}")
            return None

    def exists(self) -> bool:
        """Check if a saved state file exists."""
        return os.path.exists(self.state_path)

    def create(self, task_id: str, task_instruction: str, subtasks: List[SubTask]) -> TaskState:
        """Create a new TaskState, save it, and return it."""
        state = TaskState(
            task_id=task_id,
            task_instruction=task_instruction,
            subtasks=subtasks,
        )
        self.save(state)
        return state
