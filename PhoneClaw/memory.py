"""User memory / profile for PhoneClaw interactive mode.

Persists a JSON file at PhoneClaw/data/user_profile.json that accumulates
knowledge across sessions:

  - Task history  – what was done, what answers were found
  - User profile  – inferred name, location, language
  - App statistics – how often each app is used
  - Insights      – facts/habits extracted by LLM from completed tasks
  - Patterns      – frequently-used task types

The profile is loaded once at session start and injected into the Planner
prompt so the LLM can make more informed decisions (e.g. knows the user's
city, preferred apps, or past answers to similar questions).

After every completed task the module calls the VLM to extract any new
user insights and stores them in the profile.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PROFILE_DIR = Path(__file__).parent / "data"
DEFAULT_PROFILE_PATH = DEFAULT_PROFILE_DIR / "user_profile.json"
SCHEMA_VERSION = 1

# Maximum number of task-history entries kept in the file
MAX_TASK_HISTORY = 200
# Maximum number of insight entries kept
MAX_INSIGHTS = 100


# ---------------------------------------------------------------------------
# UserMemory
# ---------------------------------------------------------------------------

class UserMemory:
    """
    Persistent user profile and task history.

    Typical usage::

        memory = UserMemory()               # load or create profile
        memory.start_session()              # increment session counter

        # … before planning …
        context = memory.get_planner_context()   # inject into planner prompt

        # … after task …
        task_id = memory.record_task(...)
        memory.extract_insights(task, answer, task_id, agent)
    """

    def __init__(self, profile_path: Optional[str] = None):
        self.path = Path(profile_path) if profile_path else DEFAULT_PROFILE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    # ------------------------------------------------------------------
    # Load / save
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    raw = json.load(f)
                # Migrate older schema if needed
                if raw.get("schema_version", 0) < SCHEMA_VERSION:
                    raw = self._migrate(raw)
                return raw
            except Exception as exc:
                print(f"[Memory] Warning: could not load profile ({exc}). Starting fresh.")
        return self._empty_profile()

    def save(self) -> None:
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def _migrate(self, old: dict) -> dict:
        """Best-effort migration from older schema versions."""
        fresh = self._empty_profile()
        # Copy over any keys that still exist in the new schema
        for k in fresh:
            if k in old:
                fresh[k] = old[k]
        fresh["schema_version"] = SCHEMA_VERSION
        return fresh

    def _empty_profile(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "stats": {
                "total_sessions": 0,
                "total_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,
            },
            "profile": {
                "inferred_name": None,
                "inferred_location": None,
                "primary_language": "zh-CN",
                "timezone_hint": None,
                "notes": [],
            },
            "app_usage": {},
            "task_history": [],
            "insights": [],
            "frequent_patterns": {},
        }

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        """Call once at the very start of an interactive session."""
        self.data["stats"]["total_sessions"] += 1
        self.save()

    # ------------------------------------------------------------------
    # Task recording
    # ------------------------------------------------------------------

    def record_task(
        self,
        task: str,
        status: str,
        final_answer: Optional[str],
        apps_used: list[str],
        rounds: int,
        duration_seconds: float,
    ) -> int:
        """
        Append a completed task to history and update counters.

        Returns:
            task_id — 1-based sequential ID for this task entry.
        """
        stats = self.data["stats"]
        stats["total_tasks"] += 1
        if status == "completed":
            stats["completed_tasks"] += 1
        else:
            stats["failed_tasks"] += 1

        # Update per-app usage counters
        now = datetime.now().isoformat()
        for app in apps_used:
            entry = self.data["app_usage"].setdefault(app, {"count": 0, "last_used": None})
            entry["count"] += 1
            entry["last_used"] = now

        history: list = self.data["task_history"]
        task_id = len(history) + 1
        history.append({
            "id": task_id,
            "timestamp": now,
            "task": task,
            "status": status,
            "final_answer": final_answer,
            "apps_used": apps_used,
            "rounds": rounds,
            "duration_seconds": round(duration_seconds, 1),
        })

        # Trim to keep file size manageable
        if len(history) > MAX_TASK_HISTORY:
            self.data["task_history"] = history[-MAX_TASK_HISTORY:]

        self.save()
        return task_id

    def add_insight(
        self,
        text: str,
        source_task_id: int,
        confidence: str = "medium",
    ) -> bool:
        """Store a single insight string.

        Performs semantic deduplication via embedding cosine similarity
        (falls back to token-level Jaccard when the embedding API is
        unavailable).  If a semantically equivalent insight already exists
        its ``reinforced`` counter is incremented and its confidence may be
        upgraded; no new entry is created.

        Returns:
            True if the insight was new and added; False if it was a duplicate.
        """
        from PhoneClaw.embeddings import is_semantic_duplicate

        text = text.strip()
        if not text:
            return False

        existing = self.data["insights"]
        existing_texts = [i["text"] for i in existing]

        dup_idx = is_semantic_duplicate(text, existing_texts)
        if dup_idx >= 0:
            entry = existing[dup_idx]
            entry["reinforced"] = entry.get("reinforced", 1) + 1
            _conf_weight = {"high": 3, "medium": 2, "low": 1}
            if _conf_weight.get(confidence, 0) > _conf_weight.get(
                entry.get("confidence", "medium"), 0
            ):
                entry["confidence"] = confidence
            entry["last_seen"] = datetime.now().isoformat()
            self.save()
            return False

        self.data["insights"].append({
            "text": text,
            "confidence": confidence,
            "source_task_id": source_task_id,
            "timestamp": datetime.now().isoformat(),
            "reinforced": 1,
        })

        if len(self.data["insights"]) > MAX_INSIGHTS:
            self.data["insights"] = self.data["insights"][-MAX_INSIGHTS:]

        self.save()
        return True

    # ------------------------------------------------------------------
    # LLM-powered insight extraction
    # ------------------------------------------------------------------

    def extract_insights(
        self,
        task: str,
        final_answer: Optional[str],
        task_id: int,
        agent,
    ) -> list[str]:
        """
        Ask the VLM to extract user-relevant facts from a completed task.

        The agent must implement: agent.act(messages: list[dict]) -> str

        Returns:
            List of new insight strings that were added to the profile.
        """
        from PhoneClaw.prompts import MEMORY_EXTRACT_SYSTEM_PROMPT, MEMORY_EXTRACT_USER_TEMPLATE

        context_parts = [f"Task: {task}"]
        if final_answer:
            context_parts.append(f"Result/Answer: {final_answer}")
        task_context = "\n".join(context_parts)

        existing_summary = self._existing_profile_summary()

        user_content = MEMORY_EXTRACT_USER_TEMPLATE.format(
            task_context=task_context,
            existing_profile=existing_summary,
        )

        messages = [
            {"role": "system", "content": MEMORY_EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            response = agent.act(messages)
            raw_insights = self._parse_insights(response)

            added: list[str] = []
            for text in raw_insights:
                if self.add_insight(text, source_task_id=task_id):
                    added.append(text)

            if added:
                print(f"[Memory] +{len(added)} new insight(s) extracted.")
                for ins in added:
                    print(f"  • {ins}")
            return added

        except Exception as exc:
            print(f"[Memory] Could not extract insights: {exc}")
            return []

    # ------------------------------------------------------------------
    # Memory-first retrieval
    # ------------------------------------------------------------------

    def query(self, question: str, agent) -> tuple[bool, Optional[str]]:
        """
        Check whether the user's profile / task history already contains
        a confident answer to *question*.  If yes, return it immediately
        so the caller can skip device interaction entirely.

        The agent must implement: agent.act(messages: list[dict]) -> str

        Returns:
            (can_answer, answer)
            can_answer – True if the memory contains a reliable answer
            answer     – the answer string, or None when can_answer is False
        """
        profile_text = self._build_full_profile_text()
        if not profile_text:
            return False, None

        from PhoneClaw.prompts import (
            MEMORY_QUERY_SYSTEM_PROMPT,
            MEMORY_QUERY_USER_TEMPLATE,
        )

        user_content = MEMORY_QUERY_USER_TEMPLATE.format(
            question=question,
            profile=profile_text,
        )

        messages = [
            {"role": "system", "content": MEMORY_QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            response = agent.act(messages)
            can_answer, answer = self._parse_query_response(response)
            return can_answer, answer
        except Exception as exc:
            print(f"[Memory] Query error: {exc}")
            return False, None

    def _parse_query_response(self, response: str) -> tuple[bool, Optional[str]]:
        """Parse the VLM's JSON response for a memory query."""
        # Try strict JSON parse
        try:
            start = response.index("{")
            end = response.rindex("}") + 1
            data = json.loads(response[start:end])
            can_answer = bool(data.get("can_answer", False))
            answer = data.get("answer") or None
            if can_answer and answer:
                return True, str(answer).strip()
            return False, None
        except (ValueError, json.JSONDecodeError):
            pass

        # Fallback: if the model returned NOT_FOUND as plain text, treat as miss
        if "NOT_FOUND" in response or "无法回答" in response or "not found" in response.lower():
            return False, None

        # If there is a non-trivial response that doesn't look like a refusal,
        # treat it as a direct answer (last-resort heuristic)
        stripped = response.strip()
        if len(stripped) > 10:
            return True, stripped

        return False, None

    def _build_full_profile_text(self) -> str:
        """
        Build a comprehensive text dump of everything in the profile.

        This is what the VLM reads when deciding whether the memory can
        answer a question without touching the device.
        """
        parts: list[str] = []

        p = self.data["profile"]
        if p.get("inferred_name"):
            parts.append(f"User name: {p['inferred_name']}")
        if p.get("inferred_location"):
            parts.append(f"Location: {p['inferred_location']}")
        if p.get("primary_language"):
            parts.append(f"Language preference: {p['primary_language']}")
        if p.get("notes"):
            parts.append("Notes: " + "; ".join(p["notes"]))

        # All insights (the richest source)
        if self.data["insights"]:
            parts.append("\n## User Facts & Insights")
            for ins in self.data["insights"]:
                parts.append(f"  - {ins['text']}")

        # Task history with recorded answers
        answered = [t for t in self.data["task_history"] if t.get("final_answer")]
        if answered:
            parts.append("\n## Past Task Answers")
            for t in answered[-50:]:
                ts = t["timestamp"][:10]
                parts.append(f"  [{ts}] Q: {t['task']}")
                parts.append(f"         A: {t['final_answer']}")

        # App usage stats (useful for "which apps do I use most?" type queries)
        if self.data["app_usage"]:
            top_apps = sorted(
                self.data["app_usage"].items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            )[:10]
            parts.append("\n## App Usage")
            for app, v in top_apps:
                parts.append(f"  - {app}: {v['count']} times")

        return "\n".join(parts) if parts else ""

    def _parse_insights(self, response: str) -> list[str]:
        """Parse the VLM response into a list of insight strings."""
        # Try JSON array first
        try:
            start = response.index("[")
            end = response.rindex("]") + 1
            items = json.loads(response[start:end])
            return [str(item).strip() for item in items if str(item).strip()]
        except (ValueError, json.JSONDecodeError):
            pass

        # Fallback: extract bullet / numbered list lines
        lines = []
        for line in response.splitlines():
            stripped = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
            if len(stripped) > 8:
                lines.append(stripped)
        return lines[:10]

    def _existing_profile_summary(self) -> str:
        """Compact summary of already-known facts (helps LLM avoid duplicates)."""
        p = self.data["profile"]
        parts: list[str] = []

        if p.get("inferred_name"):
            parts.append(f"User name: {p['inferred_name']}")
        if p.get("inferred_location"):
            parts.append(f"Location: {p['inferred_location']}")
        if p.get("notes"):
            parts.append("Profile notes: " + "; ".join(p["notes"][:5]))

        top_apps = sorted(
            self.data["app_usage"].items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        )[:6]
        if top_apps:
            parts.append(
                "Frequent apps: "
                + ", ".join(f"{a}({v['count']}×)" for a, v in top_apps)
            )

        if self.data["insights"]:
            recent = self.data["insights"][-6:]
            parts.append(
                "Recent insights:\n"
                + "\n".join(f"  - {i['text']}" for i in recent)
            )

        return "\n".join(parts) if parts else "(none yet)"

    # ------------------------------------------------------------------
    # Planner context injection
    # ------------------------------------------------------------------

    def get_planner_context(self) -> str:
        """
        Return a formatted string to inject into the Planner system prompt.

        Provides background about the user so the planner can produce
        better-informed subtask lists.

        Returns empty string if the profile has no useful information yet.
        """
        p = self.data["profile"]
        lines: list[str] = []

        if p.get("inferred_name"):
            lines.append(f"- Name / handle: {p['inferred_name']}")
        if p.get("inferred_location"):
            lines.append(f"- Location: {p['inferred_location']}")
        if p.get("primary_language"):
            lines.append(f"- Primary language: {p['primary_language']}")

        top_apps = sorted(
            self.data["app_usage"].items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        )[:6]
        if top_apps:
            app_str = ", ".join(f"{a} ({v['count']}×)" for a, v in top_apps)
            lines.append(f"- Frequently used apps: {app_str}")

        recent_insights = self.data["insights"][-10:]
        if recent_insights:
            lines.append("- Known facts about this user:")
            for ins in recent_insights:
                lines.append(f"  • {ins['text']}")

        recent_tasks = self.data["task_history"][-4:]
        if recent_tasks:
            lines.append("- Recent tasks (for context):")
            for t in recent_tasks:
                icon = "✓" if t["status"] == "completed" else "✗"
                answer_hint = f" → {t['final_answer'][:60]}" if t.get("final_answer") else ""
                lines.append(f"  {icon} {t['task'][:80]}{answer_hint}")

        if not lines:
            return ""

        return "## User Profile (from memory)\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def session_banner(self) -> str:
        """Brief banner shown at the start of an interactive session."""
        stats = self.data["stats"]
        p = self.data["profile"]

        parts = [
            f"[Memory] Profile: {self.path}",
            (
                f"[Memory] Sessions: {stats['total_sessions']}  |  "
                f"Tasks: {stats['completed_tasks']} completed / "
                f"{stats['failed_tasks']} failed  |  "
                f"Insights: {len(self.data['insights'])}"
            ),
        ]

        if p.get("inferred_name"):
            parts.append(f"[Memory] User: {p['inferred_name']}")
        if p.get("inferred_location"):
            parts.append(f"[Memory] Location: {p['inferred_location']}")

        return "\n".join(parts)

    def get_profile_path(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        stats = self.data["stats"]
        return (
            f"UserMemory(tasks={stats['total_tasks']}, "
            f"insights={len(self.data['insights'])}, "
            f"path={self.path})"
        )
