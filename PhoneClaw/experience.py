"""Experience log for PhoneClaw — procedural memory.

Persists a JSON file at ~/.PhoneClaw/experience_log.json that accumulates
execution know-how across all tasks:

  lessons       – app-specific and general-purpose lessons derived from task
                  traces (successful paths, failed approaches, UI knowledge)
  reinforcement – existing lessons gain confidence each time they are re-confirmed

The experience log is used in two ways:

  1. BEFORE execution  – get_hints_for(app, subtask) injects relevant lessons
     into the Executor system prompt so the VLM avoids previously-failed
     approaches and prefers previously-successful ones.

  2. AFTER execution   – extract_and_record(task, subtask_logs, ...) calls the
     VLM to derive structured lessons from the completed task trace and stores
     them for future runs.

Lesson types
------------
  successful_navigation  – a confirmed sequence or coordinate that reaches a goal
  failed_approach        – a coordinate / action that was tried and failed
  ui_knowledge           – layout fact about an app's UI (tab positions, etc.)
  timing                 – when to wait, how long animations take, etc.
  general                – cross-app advice
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_LOG_PATH = Path(__file__).parent / "data" / "experience_log.json"
SCHEMA_VERSION = 1

# Maximum number of lessons to retain (oldest pruned first)
MAX_LESSONS = 500

# Confidence tiers and their sort weights (higher = shown first)
CONFIDENCE_WEIGHT = {"high": 3, "medium": 2, "low": 1}

# Auto-compaction threshold: compact an app's lessons when its count reaches this
COMPACT_THRESHOLD = 20

# Target lesson count per app after compaction (roughly 1/3 of threshold)
COMPACT_TARGET = 8


# ---------------------------------------------------------------------------
# ExperienceLog
# ---------------------------------------------------------------------------

class ExperienceLog:
    """
    Persistent store of execution lessons for the PhoneClaw agent.

    Typical usage::

        exp = ExperienceLog()

        # before executing a subtask
        hints = exp.get_hints_for("Meituan", "Navigate to orders page")

        # after a complete task
        exp.extract_and_record(task, subtask_logs, final_answer, agent)
    """

    def __init__(self, log_path: Optional[str] = None):
        self.path = Path(log_path) if log_path else DEFAULT_LOG_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    raw = json.load(f)
                if raw.get("schema_version", 0) < SCHEMA_VERSION:
                    raw = self._migrate(raw)
                return raw
            except Exception as exc:
                print(f"[Experience] Warning: could not load log ({exc}). Starting fresh.")
        return self._empty_log()

    def save(self) -> None:
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def _empty_log(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "stats": {
                "total_lessons": 0,
                "tasks_processed": 0,
                "compactions": 0,
            },
            "compaction_history": [],
            "lessons": [],
        }

    def _migrate(self, old: dict) -> dict:
        fresh = self._empty_log()
        for k in fresh:
            if k in old:
                fresh[k] = old[k]
        fresh["schema_version"] = SCHEMA_VERSION
        return fresh

    # ------------------------------------------------------------------
    # Adding lessons
    # ------------------------------------------------------------------

    def add_lesson(
        self,
        app: str,
        lesson_type: str,
        description: str,
        source_task: str,
        confidence: str = "medium",
    ) -> bool:
        """Store a single lesson.

        Performs semantic deduplication scoped to the same app: uses
        embedding cosine similarity (falls back to token-level Jaccard when
        the embedding API is unavailable).  If a semantically equivalent
        lesson already exists its ``reinforced`` counter is incremented and
        confidence may be upgraded; no new entry is created.

        Returns:
            True if new lesson was created; False if an existing one was reinforced.
        """
        from PhoneClaw.embeddings import is_semantic_duplicate

        description = description.strip()
        lessons: list = self.data["lessons"]

        # Restrict comparison to same-app lessons (bundle-ID-aware)
        same_app = [l for l in lessons if self._app_matches(app, l.get("app"))]
        same_app_texts = [l["description"] for l in same_app]

        dup_idx = is_semantic_duplicate(description, same_app_texts)
        if dup_idx >= 0:
            existing = same_app[dup_idx]
            existing["reinforced"] = existing.get("reinforced", 1) + 1
            if (
                CONFIDENCE_WEIGHT.get(confidence, 0)
                > CONFIDENCE_WEIGHT.get(existing["confidence"], 0)
            ):
                existing["confidence"] = confidence
            existing["last_seen"] = datetime.now().isoformat()
            self.save()
            return False

        # New lesson
        self.data["lessons"].append({
            "id": self.data["stats"]["total_lessons"] + 1,
            "app": app,
            "lesson_type": lesson_type,
            "description": description,
            "source_task": source_task[:120],
            "confidence": confidence,
            "reinforced": 1,
            "timestamp": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
        })
        self.data["stats"]["total_lessons"] += 1

        # Prune oldest low-confidence lessons when over limit
        if len(self.data["lessons"]) > MAX_LESSONS:
            self.data["lessons"].sort(
                key=lambda x: (
                    CONFIDENCE_WEIGHT.get(x["confidence"], 1),
                    x.get("reinforced", 1),
                )
            )
            self.data["lessons"] = self.data["lessons"][-(MAX_LESSONS):]

        self.save()
        return True

    # ------------------------------------------------------------------
    # Hint injection for the Executor
    # ------------------------------------------------------------------

    @staticmethod
    def _app_matches(query: Optional[str], lesson_app: Optional[str]) -> bool:
        """Return True when *query* and *lesson_app* refer to the same iOS app.

        Matching is attempted in two ways:
        1. Case-insensitive name equality  (fast, no imports)
        2. Same bundle ID via APP_PACKAGES_IOS  (handles Chinese ↔ English
           aliases, e.g. "Xiaohongshu" == "小红书" because both map to
           "com.xingin.discover")
        """
        if not query or not lesson_app:
            return False
        if query.lower() == lesson_app.lower():
            return True
        try:
            from PhoneClaw.actions import APP_PACKAGES_IOS
            query_bundle = APP_PACKAGES_IOS.get(query, "")
            lesson_bundle = APP_PACKAGES_IOS.get(lesson_app, "")
            if query_bundle and lesson_bundle and query_bundle == lesson_bundle:
                return True
        except Exception:
            pass
        return False

    def get_hints_for(
        self,
        app_name: Optional[str],
        subtask_instruction: str,
        max_hints: int = 8,
    ) -> str:
        """
        Return a formatted ``## Experience Notes`` block to append to the
        Executor system prompt.

        Selects lessons that are relevant to *app_name* and/or the keywords
        in *subtask_instruction*.  Returns an empty string when there are no
        applicable lessons.

        App matching is bundle-ID-aware: English and Chinese aliases for the
        same app are treated as identical (e.g. "Xiaohongshu" matches lessons
        stored under "小红书").
        """
        if not self.data["lessons"]:
            return ""

        candidates: list[dict] = []

        # Keywords from the subtask for lightweight relevance filtering
        keywords = set(re.findall(r'\w+', subtask_instruction.lower()))

        for lesson in self.data["lessons"]:
            score = 0

            # App match — uses bundle-ID comparison to handle Chinese/English aliases
            if self._app_matches(app_name, lesson.get("app")):
                score += 4
            elif lesson.get("app") in ("", "general", None):
                score += 1

            # Keyword overlap with lesson description
            lesson_words = set(re.findall(r'\w+', lesson["description"].lower()))
            overlap = len(keywords & lesson_words)
            score += overlap

            # Confidence and reinforcement boost
            score += CONFIDENCE_WEIGHT.get(lesson["confidence"], 1)
            score += min(lesson.get("reinforced", 1) - 1, 3)  # cap bonus at 3

            if score >= 4:
                candidates.append((score, lesson))

        if not candidates:
            return ""

        # Sort by score descending, take top N
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = [item for _, item in candidates[:max_hints]]

        lines = ["\n## Experience Notes (from past executions — use these to avoid repeating known mistakes)"]

        # Group by lesson type for readability
        successes = [l for l in top if l["lesson_type"] == "successful_navigation"]
        failures  = [l for l in top if l["lesson_type"] == "failed_approach"]
        ui_facts  = [l for l in top if l["lesson_type"] == "ui_knowledge"]
        others    = [l for l in top if l["lesson_type"] not in (
            "successful_navigation", "failed_approach", "ui_knowledge"
        )]

        if successes:
            lines.append("✓ What has worked before:")
            for l in successes:
                conf = f"[{l['confidence']}, confirmed {l['reinforced']}×]"
                lines.append(f"  • {l['description']}  {conf}")

        if failures:
            lines.append("✗ What has FAILED before — do NOT repeat:")
            for l in failures:
                conf = f"[seen {l['reinforced']}×]"
                lines.append(f"  • {l['description']}  {conf}")

        if ui_facts:
            lines.append("ℹ UI knowledge:")
            for l in ui_facts:
                lines.append(f"  • {l['description']}")

        for l in others:
            lines.append(f"  • {l['description']}")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # VLM-powered lesson extraction
    # ------------------------------------------------------------------

    def extract_and_record(
        self,
        task: str,
        subtask_logs: list[dict],
        final_answer: Optional[str],
        agent,
    ) -> list[str]:
        """
        Ask the VLM to derive structured lessons from the task execution trace,
        then store each lesson.

        Args:
            task:          The original task instruction.
            subtask_logs:  List of per-subtask dicts built by RalphLoop (see loop.py).
            final_answer:  Final answer / outcome, if any.
            agent:         VLM agent with act(messages) -> str.

        Returns:
            List of newly added lesson description strings.
        """
        from PhoneClaw.prompts import (
            EXPERIENCE_EXTRACT_SYSTEM_PROMPT,
            EXPERIENCE_EXTRACT_USER_TEMPLATE,
        )

        trace_summary = self._build_trace_summary(task, subtask_logs, final_answer)
        if not trace_summary.strip():
            return []

        user_content = EXPERIENCE_EXTRACT_USER_TEMPLATE.format(
            trace_summary=trace_summary,
        )

        messages = [
            {"role": "system", "content": EXPERIENCE_EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            response = agent.act(messages)
            raw_lessons = self._parse_lessons_response(response)
        except Exception as exc:
            print(f"[Experience] Could not extract lessons: {exc}")
            return []

        self.data["stats"]["tasks_processed"] += 1

        added: list[str] = []
        for item in raw_lessons:
            app       = item.get("app", "general") or "general"
            ltype     = item.get("lesson_type", "general")
            desc      = str(item.get("description", "")).strip()
            conf      = item.get("confidence", "medium")

            if not desc or len(desc) < 8:
                continue

            # Normalise lesson type
            if ltype not in (
                "successful_navigation", "failed_approach",
                "ui_knowledge", "timing", "general"
            ):
                ltype = "general"

            is_new = self.add_lesson(
                app=app,
                lesson_type=ltype,
                description=desc,
                source_task=task,
                confidence=conf,
            )
            added.append(desc)
            status = "new" if is_new else "reinforced"
            print(f"[Experience] {status}: [{app}] {desc[:80]}")

        # Auto-compact any app that has accumulated too many lessons
        self.compact_if_needed(agent)

        return added

    def _build_trace_summary(
        self,
        task: str,
        subtask_logs: list[dict],
        final_answer: Optional[str],
    ) -> str:
        """Format the execution trace into readable text for the VLM."""
        lines = [f"Task: {task}\n"]

        for entry in subtask_logs:
            status = "PASSED" if entry.get("passed") else "FAILED"
            app_tag = f" [app: {entry['app']}]" if entry.get("app") else ""
            lines.append(f"Subtask{app_tag}: {entry['instruction']}")
            lines.append(f"  Outcome: {status}")

            actions = entry.get("actions", [])
            if actions:
                for act in actions:
                    result_icon = "✓" if act.get("passed") else "✗"
                    reason = act.get("reason", "")[:100]
                    lines.append(f"    {result_icon} {act['action']}  → {reason}")
            lines.append("")

        if final_answer:
            lines.append(f"Final answer: {final_answer}")

        return "\n".join(lines)

    def _parse_lessons_response(self, response: str) -> list[dict]:
        """Parse the VLM's JSON array response into lesson dicts."""
        try:
            start = response.index("[")
            end = response.rindex("]") + 1
            items = json.loads(response[start:end])
            if isinstance(items, list):
                return [i for i in items if isinstance(i, dict)]
        except (ValueError, json.JSONDecodeError):
            pass
        return []

    # ------------------------------------------------------------------
    # Compaction: consolidate redundant lessons via VLM
    # ------------------------------------------------------------------

    def compact_if_needed(
        self,
        agent,
        threshold: int = COMPACT_THRESHOLD,
        target: int = COMPACT_TARGET,
    ) -> list[str]:
        """Check each app's lesson count and compact any that exceed *threshold*.

        Compaction calls the VLM to merge near-duplicate lessons, remove
        low-value entries (e.g. individual keystrokes), and generalise
        coordinates — reducing storage and improving hint quality.

        Args:
            agent:     VLM agent with act(messages) -> str.
            threshold: Compact an app when it has at least this many lessons.
            target:    Desired lesson count per app after compaction.

        Returns:
            List of app names that were compacted.
        """
        # Count per-app lessons
        app_counts: dict[str, int] = {}
        for lesson in self.data["lessons"]:
            app = lesson.get("app") or "general"
            app_counts[app] = app_counts.get(app, 0) + 1

        compacted: list[str] = []
        for app, count in app_counts.items():
            if count >= threshold:
                print(
                    f"[Experience] '{app}' has {count} lessons "
                    f"(threshold={threshold}) — compacting..."
                )
                n_before, n_after = self.compact_app_lessons(
                    app_name=app, agent=agent, target=target
                )
                if n_after < n_before:
                    compacted.append(app)
                    print(
                        f"[Experience] '{app}' compacted: "
                        f"{n_before} → {n_after} lessons"
                    )
                else:
                    print(
                        f"[Experience] '{app}' compaction returned no improvement."
                    )

        return compacted

    def compact_app_lessons(
        self,
        app_name: str,
        agent,
        target: int = COMPACT_TARGET,
    ) -> tuple[int, int]:
        """Use the VLM to consolidate all lessons for *app_name* into a
        compact, high-quality set.

        The raw lessons are replaced in-place with the consolidated output.
        A record is appended to ``compaction_history``.

        Args:
            app_name: Name of the app whose lessons to compact.
            agent:    VLM agent with act(messages) -> str.
            target:   Desired lesson count after compaction.

        Returns:
            (n_before, n_after) lesson counts.
        """
        from PhoneClaw.prompts import (
            EXPERIENCE_COMPACT_SYSTEM_PROMPT,
            EXPERIENCE_COMPACT_USER_TEMPLATE,
        )

        app_lessons = [
            l for l in self.data["lessons"]
            if self._app_matches(app_name, l.get("app"))
        ]
        other_lessons = [
            l for l in self.data["lessons"]
            if not self._app_matches(app_name, l.get("app"))
        ]

        n_before = len(app_lessons)
        if n_before == 0:
            return 0, 0

        # Build a compact representation to send to VLM (omit internal fields)
        lessons_for_vlm = [
            {
                "description": l["description"],
                "lesson_type": l["lesson_type"],
                "confidence": l["confidence"],
                "reinforced": l.get("reinforced", 1),
            }
            for l in app_lessons
        ]

        system_content = EXPERIENCE_COMPACT_SYSTEM_PROMPT.replace(
            "{target_count}", str(target)
        )
        user_content = EXPERIENCE_COMPACT_USER_TEMPLATE.format(
            app_name=app_name,
            lesson_count=n_before,
            target_count=target,
            lessons_json=json.dumps(lessons_for_vlm, ensure_ascii=False, indent=2),
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        try:
            response = agent.act(messages)
            compact_items = self._parse_lessons_response(response)
        except Exception as exc:
            print(f"[Experience] Compaction VLM call failed for '{app_name}': {exc}")
            return n_before, n_before

        if not compact_items:
            print(f"[Experience] VLM returned no items for '{app_name}' — keeping original.")
            return n_before, n_before

        now = datetime.now().isoformat()
        new_lessons: list[dict] = []
        for item in compact_items:
            desc = str(item.get("description", "")).strip()
            if not desc:
                continue
            new_lessons.append({
                "id": self.data["stats"]["total_lessons"] + len(new_lessons) + 1,
                "app": app_name,
                "lesson_type": item.get("lesson_type", "general"),
                "description": desc,
                "source_task": "compaction",
                "confidence": item.get("confidence", "medium"),
                "reinforced": max(1, int(item.get("reinforced", 1))),
                "timestamp": now,
                "last_seen": now,
                "compacted": True,
            })

        # Replace app lessons with compacted set
        self.data["lessons"] = other_lessons + new_lessons

        # Update stats
        self.data["stats"].setdefault("compactions", 0)
        self.data["stats"]["compactions"] += 1
        self.data["stats"]["total_lessons"] = len(self.data["lessons"])

        # Record history entry
        self.data.setdefault("compaction_history", []).append({
            "app": app_name,
            "before": n_before,
            "after": len(new_lessons),
            "timestamp": now,
        })

        self.save()
        return n_before, len(new_lessons)

    def compact_all(self, agent, target: int = COMPACT_TARGET) -> dict[str, tuple[int, int]]:
        """Compact lessons for ALL apps regardless of lesson count.

        Useful for a one-off cleanup of an existing log that has accumulated
        many redundant entries.

        Returns:
            Dict mapping app_name → (n_before, n_after).
        """
        apps = list({
            (l.get("app") or "general")
            for l in self.data["lessons"]
        })
        results: dict[str, tuple[int, int]] = {}
        for app in apps:
            print(f"[Experience] Compacting all lessons for '{app}'...")
            results[app] = self.compact_app_lessons(app, agent, target=target)
        return results

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def summary_banner(self) -> str:
        stats = self.data["stats"]
        return (
            f"[Experience] Log: {self.path}  |  "
            f"Lessons: {stats['total_lessons']}  |  "
            f"Tasks processed: {stats['tasks_processed']}"
        )

    def get_lessons_for_app(self, app_name: str) -> list[dict]:
        """Return all lessons for a given app, sorted by confidence.

        App matching is bundle-ID-aware so that English and Chinese aliases
        (e.g. "Xiaohongshu" and "小红书") return the same set of lessons.
        """
        return sorted(
            [l for l in self.data["lessons"] if self._app_matches(app_name, l.get("app"))],
            key=lambda x: (
                CONFIDENCE_WEIGHT.get(x["confidence"], 1),
                x.get("reinforced", 1),
            ),
            reverse=True,
        )

    def __repr__(self) -> str:
        stats = self.data["stats"]
        return (
            f"ExperienceLog(lessons={stats['total_lessons']}, "
            f"tasks={stats['tasks_processed']}, path={self.path})"
        )
