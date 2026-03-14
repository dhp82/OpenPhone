#!/usr/bin/env python3
"""
PhoneClaw: iOS GUI Agent based on Ralph Loop

--- Single-task mode ---

    export OPENROUTER_API_KEY="sk-or-v1-..."

    python PhoneClaw/run_phoneclaw.py \\
        --task "打开微信，查看最近5条对话" \\
        --openrouter \\
        --model-name "z-ai/glm-4.6v"

--- Interactive / daemon mode ---

    python PhoneClaw/run_phoneclaw.py \\
        --interactive \\
        --openrouter \\
        --model-name "z-ai/glm-4.6v"

    Connects once, then waits for tasks typed at the prompt.
    The device screen is kept awake automatically (--keepalive-interval controls
    how often a ping is sent; default: 30 s).
    Type 'quit' or press Ctrl+C to exit.

--- Environment variables ---

    # OpenRouter (preferred)
    OPENROUTER_API_KEY        - Your OpenRouter API key
    OPENROUTER_MODEL          - Default executor model on OpenRouter
    EVAL_OPENROUTER_MODEL     - Evaluator model (falls back to OPENROUTER_MODEL)

    # Local / generic OpenAI-compatible (fallback)
    WDA_URL                   - WebDriverAgent URL (default: http://localhost:8100)
    API_BASE                  - Executor VLM endpoint
    MODEL_NAME                - Executor VLM model name
    API_KEY                   - API key (default: EMPTY for local models)
    AGENT_TYPE                - "OpenAIAgent" or "QwenVLAgent"

    EVAL_API_BASE             - Evaluator VLM endpoint (falls back to API_BASE)
    EVAL_MODEL_NAME           - Evaluator model name (falls back to MODEL_NAME)
    EVAL_API_KEY              - Evaluator API key (falls back to API_KEY)
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Allow running from the Android-Lab root
sys.path.insert(0, str(Path(__file__).parent.parent))

from PhoneClaw.connection import IOSConnection
from PhoneClaw.controller import IOSController
from PhoneClaw.executor import IOSExecutor
from PhoneClaw.recorder import PhoneClawRecorder
from PhoneClaw.state import StateManager
from PhoneClaw.planner import TaskPlanner
from PhoneClaw.evaluator import SubTaskEvaluator
from PhoneClaw.loop import RalphLoop
from PhoneClaw.agent import OpenRouterAgent, OPENROUTER_BASE_URL
from PhoneClaw.keepalive import ScreenKeepalive
from PhoneClaw.memory import UserMemory
from PhoneClaw.experience import ExperienceLog
from PhoneClaw.learn import DemoRecorder


class MobileClawConfig:
    """Simple config holder for PhoneClaw agent."""

    def __init__(self, task_dir: str):
        self.task_dir = task_dir
        self.screenshot_dir = os.path.join(task_dir, "screenshots")


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_agent(
    *,
    use_openrouter: bool,
    api_key: str,
    model_name: str,
    api_base: str,
    agent_type: str,
    openrouter_site_url: str,
    openrouter_app_title: str,
):
    """
    Instantiate the appropriate VLM agent.

    When use_openrouter=True returns an OpenRouterAgent from PhoneClaw.agent.
    Otherwise falls back to the Android-Lab agent classes (OpenAIAgent / QwenVLAgent).
    """
    if use_openrouter:
        return OpenRouterAgent(
            api_key=api_key,
            model_name=model_name,
            api_base=api_base,
            site_url=openrouter_site_url,
            app_title=openrouter_app_title,
        )

    from agent.model import QwenVLAgent, OpenAIAgent
    if agent_type == "QwenVLAgent":
        return QwenVLAgent(api_key=api_key, api_base=api_base, model_name=model_name)
    else:
        return OpenAIAgent(api_key=api_key, api_base=api_base, model_name=model_name)


# ---------------------------------------------------------------------------
# Per-task execution (re-entrant: may be called multiple times in daemon mode)
# ---------------------------------------------------------------------------

def _run_single_task(
    task_instruction: str,
    args,
    controller: IOSController,
    executor: IOSExecutor,
    exec_agent,
    eval_agent,
    planner: TaskPlanner,
    evaluator: SubTaskEvaluator,
    task_dir_override: Optional[str] = None,
    resume: bool = False,
    memory: Optional[UserMemory] = None,
    experience: Optional[ExperienceLog] = None,
) -> None:
    """
    Plan and execute one complete task.

    Creates a fresh log directory, recorder and state manager for each task
    so runs are fully independent when called repeatedly from the interactive
    loop.

    If a UserMemory instance is provided, the task result is recorded and
    the VLM is called to extract new insights about the user.
    """
    start_time = time.time()

    # ------------------------------------------------------------------
    # Memory-first retrieval: if the profile already has a confident
    # answer to this question, return it immediately without touching
    # the device.  Only skipped when --resume is active (the caller
    # explicitly wants to re-run a device task).
    # ------------------------------------------------------------------
    if memory is not None and not resume:
        print("[Memory] Checking profile for cached answer...")
        mem_can_answer, mem_answer = memory.query(task_instruction, exec_agent)
        if mem_can_answer and mem_answer:
            print("[Memory] Answer found in profile — skipping device interaction.\n")
            print("=" * 60)
            print("[PhoneClaw] ANSWER  (from memory)")
            print("=" * 60)
            print(mem_answer)
            print("=" * 60 + "\n")
            # Record this as a zero-round completed task so history stays consistent
            memory.record_task(
                task=task_instruction,
                status="completed",
                final_answer=mem_answer,
                apps_used=[],
                rounds=0,
                duration_seconds=round(time.time() - start_time, 1),
            )
            return
        else:
            print("[Memory] Not in profile — will use device.\n")

    timestamp = int(start_time)
    dt_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H-%M-%S")
    task_id = f"phoneclaw_{timestamp}"

    task_dir = task_dir_override or f"./phoneclaw_logs/{task_id}_{dt_str}"
    config = MobileClawConfig(task_dir=task_dir)
    os.makedirs(config.task_dir, exist_ok=True)
    os.makedirs(config.screenshot_dir, exist_ok=True)

    # Point device components at this task's screenshot directory
    controller.screenshot_dir = config.screenshot_dir
    executor.screenshot_dir = config.screenshot_dir

    recorder = PhoneClawRecorder(
        task_id=task_id,
        instruction=task_instruction,
        page_executor=executor,
        config=config,
    )
    state_manager = StateManager(state_dir=config.task_dir)

    state = None
    if resume and state_manager.exists():
        state = state_manager.load()
        if state:
            print(
                f"[PhoneClaw] Resuming from saved state. "
                f"Subtask {state.current_subtask_idx + 1}/{len(state.subtasks)}, "
                f"Round {state.round_count}"
            )
        else:
            print("[PhoneClaw] Warning: Failed to load saved state. Starting fresh.")

    if state is None:
        print("[PhoneClaw] Planning subtasks...")
        # Inject user context into the planner when memory is available
        user_context = memory.get_planner_context() if memory else ""
        subtasks = planner.plan(task_instruction, user_context=user_context)
        state = state_manager.create(
            task_id=task_id,
            task_instruction=task_instruction,
            subtasks=subtasks,
        )

    ralph = RalphLoop(
        controller=controller,
        executor=executor,
        agent=exec_agent,
        evaluator=evaluator,
        recorder=recorder,
        state_manager=state_manager,
        max_rounds=args.max_rounds,
        max_fix_retries=args.max_fix_retries,
        request_interval=args.request_interval,
        skip_failed_subtasks=not args.no_skip_failed,
        experience=experience,
    )

    print(f"\n[PhoneClaw] Task: {task_instruction}")
    print(f"[PhoneClaw] Logs: {config.task_dir}\n")

    final_state = ralph.run(state)
    duration = time.time() - start_time

    print("\n" + "=" * 60)
    if final_state.status == "completed":
        print("[PhoneClaw] Task COMPLETED successfully.")
    else:
        print(f"[PhoneClaw] Task ended with status: {final_state.status}")
    print("=" * 60)
    print(final_state.summary())
    print(f"\nFull trace saved to: {recorder.trace_file_path}")

    # ------------------------------------------------------------------
    # Memory: record task and extract insights
    # ------------------------------------------------------------------
    if memory is not None:
        import re as _re

        # Extract app names mentioned in launch() calls across all subtasks
        apps_used: list[str] = []
        for subtask in final_state.subtasks:
            for match in _re.findall(r'launch\("([^"]+)"\)', subtask.instruction):
                if match not in apps_used:
                    apps_used.append(match)

        mem_task_id = memory.record_task(
            task=task_instruction,
            status=final_state.status,
            final_answer=ralph.last_final_answer,
            apps_used=apps_used,
            rounds=final_state.round_count,
            duration_seconds=duration,
        )

        # Background insight extraction (same process, non-blocking in most cases)
        memory.extract_insights(
            task=task_instruction,
            final_answer=ralph.last_final_answer,
            task_id=mem_task_id,
            agent=exec_agent,
        )


# ---------------------------------------------------------------------------
# Interactive / daemon loop
# ---------------------------------------------------------------------------

def _run_interactive_loop(
    args,
    controller: IOSController,
    executor: IOSExecutor,
    exec_agent,
    eval_agent,
    planner: TaskPlanner,
    evaluator: SubTaskEvaluator,
) -> None:
    """
    REPL loop: accept tasks from stdin and execute them one at a time.

    The loop runs until the user types 'quit' / 'exit' or presses Ctrl+C.
    Between tasks the device remains connected and the keepalive thread
    (started by main()) keeps the screen awake.

    A UserMemory instance is created once per session and passed to every
    task run so that task history, user insights, and app preferences
    accumulate persistently in ~/.PhoneClaw/user_profile.json.
    """
    QUIT_COMMANDS = {"quit", "exit", "q", "退出", "exit()", "quit()"}
    MEMORY_COMMANDS = {"memory", "profile", "mem", "档案", "记忆"}
    EXPERIENCE_COMMANDS = {"experience", "exp", "lessons", "经验", "经验日志"}
    COMPACT_COMMANDS = {"compact", "整理", "压缩经验", "整理经验"}
    BANNER = "=" * 60

    # ------------------------------------------------------------------
    # Initialise persistent user memory
    # ------------------------------------------------------------------
    profile_path = getattr(args, "memory_path", None)
    memory = UserMemory(profile_path=profile_path) if not getattr(args, "no_memory", False) else None
    if memory:
        memory.start_session()

    # ------------------------------------------------------------------
    # Initialise experience log
    # ------------------------------------------------------------------
    exp_path = getattr(args, "experience_path", None)
    experience = (
        ExperienceLog(log_path=exp_path)
        if not getattr(args, "no_experience", False)
        else None
    )

    print(f"\n{BANNER}")
    print("[PhoneClaw] Interactive mode — device connected.")
    if memory:
        print(memory.session_banner())
    if experience:
        print(experience.summary_banner())
    print(f"\n[PhoneClaw] Enter a task and press Enter to run it.")
    print(
        "[PhoneClaw] Commands: "
        "'memory' — profile  |  "
        "'experience' — lessons  |  "
        "'compact' — consolidate experience  |  "
        "'quit' — exit"
    )
    print(f"{BANNER}\n")

    task_count = 0

    while True:
        try:
            task_instruction = input("[PhoneClaw] Task> ").strip()
        except EOFError:
            print("\n[PhoneClaw] stdin closed. Exiting.")
            break
        except KeyboardInterrupt:
            print("\n[PhoneClaw] Interrupted. Exiting interactive mode.")
            break

        if not task_instruction:
            continue

        if task_instruction.lower() in QUIT_COMMANDS:
            print("[PhoneClaw] Goodbye.")
            break

        # Special command: print current profile summary
        if task_instruction.lower() in MEMORY_COMMANDS:
            if memory:
                _print_memory_summary(memory)
            else:
                print("[PhoneClaw] Memory is disabled (--no-memory).")
            continue

        # Special command: print experience log summary
        if task_instruction.lower() in EXPERIENCE_COMMANDS:
            if experience:
                _print_experience_summary(experience)
            else:
                print("[PhoneClaw] Experience log is disabled (--no-experience).")
            continue

        # Special command: manually trigger full compaction of experience log
        if task_instruction.lower() in COMPACT_COMMANDS:
            if experience:
                print("[PhoneClaw] Running full experience compaction (may take a minute)...")
                results = experience.compact_all(agent=exec_agent)
                for app, (before, after) in results.items():
                    print(f"  [{app}]  {before} → {after} lessons")
                _print_experience_summary(experience)
            else:
                print("[PhoneClaw] Experience log is disabled (--no-experience).")
            continue

        task_count += 1
        print(f"\n[PhoneClaw] ── Task #{task_count} ──────────────────────────")

        try:
            _run_single_task(
                task_instruction=task_instruction,
                args=args,
                controller=controller,
                executor=executor,
                exec_agent=exec_agent,
                eval_agent=eval_agent,
                planner=planner,
                evaluator=evaluator,
                memory=memory,
                experience=experience,
            )
        except KeyboardInterrupt:
            print("\n[PhoneClaw] Task interrupted by user. Ready for next task.")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"\n[PhoneClaw] Task failed with error: {exc}")
            print("[PhoneClaw] Ready for next task.")


def _print_memory_summary(memory: UserMemory) -> None:
    """Pretty-print the current user profile to the terminal."""
    data = memory.data
    stats = data["stats"]
    profile = data["profile"]
    BANNER = "=" * 60

    print(f"\n{BANNER}")
    print(f"  User Profile  —  {memory.get_profile_path()}")
    print(BANNER)
    print(f"  Sessions     : {stats['total_sessions']}")
    print(f"  Tasks total  : {stats['total_tasks']}  "
          f"(✓ {stats['completed_tasks']}  ✗ {stats['failed_tasks']})")
    print(f"  Insights     : {len(data['insights'])}")

    if profile.get("inferred_name"):
        print(f"  Name         : {profile['inferred_name']}")
    if profile.get("inferred_location"):
        print(f"  Location     : {profile['inferred_location']}")
    if profile.get("primary_language"):
        print(f"  Language     : {profile['primary_language']}")

    top_apps = sorted(
        data["app_usage"].items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    )[:8]
    if top_apps:
        print("\n  App usage:")
        for app, v in top_apps:
            print(f"    {app:<20} {v['count']}×  (last: {v['last_used'][:10]})")

    if data["insights"]:
        print("\n  Insights:")
        for ins in data["insights"][-12:]:
            print(f"    • {ins['text']}")

    if data["task_history"]:
        print("\n  Recent tasks:")
        for t in data["task_history"][-5:]:
            icon = "✓" if t["status"] == "completed" else "✗"
            ts = t["timestamp"][:16]
            ans = f"  → {t['final_answer'][:50]}" if t.get("final_answer") else ""
            print(f"    {icon} [{ts}] {t['task'][:60]}{ans}")

    print(BANNER + "\n")


def _print_experience_summary(experience: ExperienceLog) -> None:
    """Pretty-print the experience log to the terminal."""
    data = experience.data
    stats = data["stats"]
    lessons = data["lessons"]
    BANNER = "=" * 60

    print(f"\n{BANNER}")
    print(f"  Experience Log  —  {experience.path}")
    print(BANNER)
    print(f"  Lessons      : {stats['total_lessons']}")
    print(f"  Tasks processed: {stats['tasks_processed']}")

    if not lessons:
        print("  (no lessons recorded yet)")
        print(BANNER + "\n")
        return

    # Group by app
    by_app: dict[str, list] = {}
    for lesson in lessons:
        app = lesson.get("app") or "general"
        by_app.setdefault(app, []).append(lesson)

    for app, app_lessons in sorted(by_app.items()):
        print(f"\n  [{app}]")
        # Sort by confidence + reinforcement
        app_lessons.sort(
            key=lambda x: (
                {"high": 3, "medium": 2, "low": 1}.get(x["confidence"], 1),
                x.get("reinforced", 1),
            ),
            reverse=True,
        )
        for l in app_lessons[:8]:
            ltype_icon = {"successful_navigation": "✓", "failed_approach": "✗",
                          "ui_knowledge": "ℹ", "timing": "⏱"}.get(l["lesson_type"], "•")
            conf = l["confidence"][0].upper()
            reinforced = l.get("reinforced", 1)
            print(f"    {ltype_icon}[{conf}×{reinforced}] {l['description'][:80]}")

    print(BANNER + "\n")


# ---------------------------------------------------------------------------
# Learning / demonstration mode
# ---------------------------------------------------------------------------

def _run_learn_mode(
    args,
    wda_url: str,
    session_id: str,
    exec_agent,
    experience: ExperienceLog,
) -> None:
    """Record a human demonstration and extract navigation lessons.

    The function:
    1. Creates a DemoRecorder tied to the given app / task description.
    2. Starts background screenshot polling (8 fps by default).
    3. Waits for the user to press Enter (or for --learn-duration seconds).
    4. Stops recording and runs VLM analysis on each captured frame.
    5. Lessons are stored in the ExperienceLog immediately.
    """
    from pathlib import Path as _Path

    app_name = args.learn_app or "unknown"
    task_desc = args.learn_describe or f"Demonstration on {app_name}"
    poll_interval: float = getattr(args, "learn_poll", 0.12)
    change_threshold: float = getattr(args, "learn_threshold", 0.003)
    duration: Optional[float] = getattr(args, "learn_duration", None)
    demo_dir_arg: Optional[str] = getattr(args, "learn_dir", None)
    demo_dir = _Path(demo_dir_arg) if demo_dir_arg else None

    recorder = DemoRecorder(
        wda_url=wda_url,
        session_id=session_id,
        app_name=app_name,
        task_description=task_desc,
        demo_dir=demo_dir,
        poll_interval=poll_interval,
        change_threshold=change_threshold,
        experience=experience,
    )

    recorder.start()

    try:
        if duration and duration > 0:
            print(
                f"[Learn] Recording for {duration:.0f} seconds "
                f"(Ctrl+C to stop early)..."
            )
            time.sleep(duration)
        else:
            input(
                "[Learn] Perform the demo on the device.\n"
                "[Learn] Press Enter when finished...\n"
            )
    except KeyboardInterrupt:
        print("\n[Learn] Recording interrupted by user.")
    finally:
        recorder.stop()

    print(recorder.summary())

    no_analyse = getattr(args, "no_analyse", False)
    if no_analyse:
        print("[Learn] Skipping VLM analysis (--no-analyse).")
        return

    lessons = recorder.analyze_and_learn(agent=exec_agent)
    if lessons:
        print("\n[Learn] Lessons added to ExperienceLog:")
        for i, lesson in enumerate(lessons, 1):
            print(f"  {i:2d}. {lesson[:100]}")
    else:
        print("[Learn] No lessons were extracted.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PhoneClaw: iOS Ralph Loop GUI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ---- Device / WDA ----
    parser.add_argument(
        "--wda-url",
        type=str,
        default=os.getenv("WDA_URL", "http://localhost:8100"),
        help="WebDriverAgent URL (default: http://localhost:8100)",
    )

    # ---- Task / mode ----
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="High-level task description to execute (omit when using --interactive)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help=(
            "Run in interactive daemon mode: connect once, then wait for tasks "
            "typed at the prompt. Screen keepalive is enabled automatically."
        ),
    )

    # ---- Learning / demonstration mode ----
    parser.add_argument(
        "--learn",
        action="store_true",
        default=False,
        help=(
            "Learning mode: record a human demonstration on the device and "
            "extract navigation lessons into the ExperienceLog. "
            "Requires --learn-app and optionally --learn-describe."
        ),
    )
    parser.add_argument(
        "--learn-app",
        type=str,
        default=None,
        metavar="APP_NAME",
        help="Name of the app being demonstrated (e.g. '美团'). Used to scope lessons.",
    )
    parser.add_argument(
        "--learn-describe",
        type=str,
        default=None,
        metavar="DESCRIPTION",
        help="Short description of the task being demonstrated (e.g. '查看历史订单').",
    )
    parser.add_argument(
        "--learn-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Automatically stop recording after this many seconds. "
            "If omitted, recording runs until you press Enter."
        ),
    )
    parser.add_argument(
        "--learn-poll",
        type=float,
        default=0.12,
        metavar="SECONDS",
        help="Seconds between screenshots during recording (default: 0.12 ≈ 8 fps).",
    )
    parser.add_argument(
        "--learn-threshold",
        type=float,
        default=0.003,
        metavar="FRACTION",
        help=(
            "Minimum fraction of pixels that must change to register an event "
            "(default: 0.003 = 0.3%%)."
        ),
    )
    parser.add_argument(
        "--learn-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Directory to save demo frames and summary (auto-generated if omitted).",
    )
    parser.add_argument(
        "--no-analyse",
        action="store_true",
        default=False,
        help="Skip the VLM analysis step after recording (frames are still saved).",
    )

    # ---- Memory / user profile ----
    parser.add_argument(
        "--memory-path",
        type=str,
        default=os.getenv("PHONECLAW_MEMORY", None),
        metavar="PATH",
        help=(
            "Path to the user profile JSON file "
            "(default: ~/.PhoneClaw/user_profile.json). "
            "Env: PHONECLAW_MEMORY"
        ),
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        default=False,
        help="Disable user memory / profile recording for this run.",
    )
    parser.add_argument(
        "--experience-path",
        type=str,
        default=os.getenv("PHONECLAW_EXPERIENCE", None),
        metavar="PATH",
        help=(
            "Path to the experience log JSON file "
            "(default: ~/.PhoneClaw/experience_log.json). "
            "Env: PHONECLAW_EXPERIENCE"
        ),
    )
    parser.add_argument(
        "--no-experience",
        action="store_true",
        default=False,
        help="Disable experience log (lesson recording and injection) for this run.",
    )

    # ---- Screen keepalive ----
    parser.add_argument(
        "--keepalive-interval",
        type=float,
        default=float(os.getenv("KEEPALIVE_INTERVAL", "30")),
        metavar="SECONDS",
        help=(
            "Seconds between screen-keepalive pings in interactive mode "
            "(default: 30). Set to 0 to disable keepalive."
        ),
    )

    # ---- Ralph Loop parameters ----
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=100,
        help="Global cap on total action rounds (default: 100)",
    )
    parser.add_argument(
        "--max-fix-retries",
        type=int,
        default=3,
        help="Max fix attempts per subtask before skipping (default: 3)",
    )
    parser.add_argument(
        "--no-skip-failed",
        action="store_true",
        help="Abort the entire task when a subtask fails (default: skip and continue)",
    )

    # ---- Logging / resume ----
    parser.add_argument(
        "--task-dir",
        type=str,
        default=None,
        help="Directory to save logs, screenshots, and state (auto-generated if omitted)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved state in --task-dir (single-task mode only)",
    )

    # ---- Timing ----
    parser.add_argument(
        "--request-interval",
        type=float,
        default=2.0,
        help="Seconds to sleep between action rounds (default: 2.0)",
    )

    # ---- OpenRouter (primary VLM backend) ----
    parser.add_argument(
        "--openrouter",
        action="store_true",
        default=bool(os.getenv("OPENROUTER_API_KEY")),
        help="Use OpenRouter as the VLM backend (auto-enabled when OPENROUTER_API_KEY is set)",
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key (env: OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--openrouter-base-url",
        type=str,
        default=OPENROUTER_BASE_URL,
        help=f"OpenRouter API base URL (default: {OPENROUTER_BASE_URL})",
    )
    parser.add_argument(
        "--openrouter-site-url",
        type=str,
        default=os.getenv("OPENROUTER_SITE_URL", "None"),
        help="HTTP-Referer header sent to OpenRouter (env: OPENROUTER_SITE_URL)",
    )
    parser.add_argument(
        "--openrouter-app-title",
        type=str,
        default=os.getenv("OPENROUTER_APP_TITLE", "PhoneClaw"),
        help="X-Title header sent to OpenRouter (env: OPENROUTER_APP_TITLE)",
    )

    # ---- Executor model ----
    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("OPENROUTER_MODEL", os.getenv("MODEL_NAME", "z-ai/glm-4.6v")),
        help=(
            "Executor VLM model name. "
            "For OpenRouter use format 'provider/model-name' (e.g. 'openai/gpt-4o'). "
            "Env: OPENROUTER_MODEL or MODEL_NAME"
        ),
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.getenv("API_BASE", "http://localhost:8002/v1"),
        help="Executor VLM API base URL (local mode only; ignored when --openrouter is set)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("API_KEY", "EMPTY"),
        help="Executor VLM API key (local mode only; use --openrouter-api-key for OpenRouter)",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default=os.getenv("AGENT_TYPE", "OpenAIAgent"),
        choices=["OpenAIAgent", "QwenVLAgent"],
        help="Executor agent class for local mode (default: OpenAIAgent)",
    )

    # ---- Evaluator model overrides ----
    parser.add_argument(
        "--eval-model-name",
        type=str,
        default=os.getenv("EVAL_OPENROUTER_MODEL", os.getenv("EVAL_MODEL_NAME", None)),
        help=(
            "Evaluator VLM model name. Falls back to --model-name if omitted. "
            "Env: EVAL_OPENROUTER_MODEL or EVAL_MODEL_NAME"
        ),
    )
    parser.add_argument(
        "--eval-api-base",
        type=str,
        default=os.getenv("EVAL_API_BASE", None),
        help="Evaluator VLM API base (local mode only; defaults to --api-base)",
    )
    parser.add_argument(
        "--eval-api-key",
        type=str,
        default=os.getenv("EVAL_API_KEY", None),
        help="Evaluator VLM API key (local mode only; defaults to --api-key)",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Validate mode / task
    # -----------------------------------------------------------------------
    if not args.interactive and not args.task and not args.learn:
        parser.error(
            "Provide --task <description>, --interactive for daemon mode, "
            "or --learn for demonstration recording mode."
        )

    if args.interactive and args.task:
        print("[PhoneClaw] Warning: --task is ignored in --interactive mode.")

    if args.learn and not args.learn_app:
        parser.error("--learn requires --learn-app <app_name>.")

    # -----------------------------------------------------------------------
    # Validate OpenRouter configuration
    # -----------------------------------------------------------------------
    if args.openrouter:
        if not args.openrouter_api_key:
            print("[PhoneClaw] ERROR: --openrouter requires an API key.")
            print("  Set it via --openrouter-api-key or export OPENROUTER_API_KEY=sk-or-v1-...")
            sys.exit(1)
        print(f"[PhoneClaw] Using OpenRouter | executor model: {args.model_name}")
        eval_model = args.eval_model_name or args.model_name
        print(f"[PhoneClaw] Using OpenRouter | evaluator model: {eval_model}")
    else:
        if not args.api_base or not args.model_name:
            print("[PhoneClaw] ERROR: Missing required environment variables for local mode.")
            print("  export API_BASE='http://localhost:8002/v1'")
            print("  export MODEL_NAME='your-model-name'")
            print("  (or use --openrouter with OPENROUTER_API_KEY for cloud models)")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # 1. Check WDA connection
    # -----------------------------------------------------------------------
    print("[PhoneClaw] Checking WebDriverAgent connection...")
    conn = IOSConnection(wda_url=args.wda_url)

    if not conn.is_wda_ready():
        print(f"[PhoneClaw] ERROR: WebDriverAgent not ready at {args.wda_url}")
        print("Please start WebDriverAgent on your iOS device first.")
        sys.exit(1)

    print("[PhoneClaw] WebDriverAgent ready.")

    success, session_id = conn.start_wda_session()
    if not success:
        print(f"[PhoneClaw] ERROR: Failed to start WDA session: {session_id}")
        sys.exit(1)

    print(f"[PhoneClaw] WDA session started: {session_id}")

    # -----------------------------------------------------------------------
    # 2. Initialise device components (shared across all tasks)
    # -----------------------------------------------------------------------
    controller = IOSController(wda_url=args.wda_url, session_id=session_id)
    executor = IOSExecutor(wda_url=args.wda_url, session_id=session_id)

    # -----------------------------------------------------------------------
    # 3. Initialise VLM agents (shared across all tasks)
    # -----------------------------------------------------------------------
    _common = dict(
        use_openrouter=args.openrouter,
        openrouter_site_url=args.openrouter_site_url,
        openrouter_app_title=args.openrouter_app_title,
    )

    print("[PhoneClaw] Loading executor agent...")
    exec_agent = _build_agent(
        api_key=args.openrouter_api_key if args.openrouter else args.api_key,
        model_name=args.model_name,
        api_base=args.openrouter_base_url if args.openrouter else args.api_base,
        agent_type=args.agent_type,
        **_common,
    )

    eval_model_name = args.eval_model_name or args.model_name
    eval_api_base = args.eval_api_base or (args.openrouter_base_url if args.openrouter else args.api_base)
    eval_api_key = args.eval_api_key or (args.openrouter_api_key if args.openrouter else args.api_key)

    print("[PhoneClaw] Loading evaluator agent...")
    eval_agent = _build_agent(
        api_key=eval_api_key,
        model_name=eval_model_name,
        api_base=eval_api_base,
        agent_type=args.agent_type,
        **_common,
    )

    planner = TaskPlanner(agent=exec_agent)
    evaluator = SubTaskEvaluator(agent=eval_agent)

    # -----------------------------------------------------------------------
    # 4. Screen keepalive
    #    Always on in interactive mode; opt-in via --keepalive-interval > 0
    #    in single-task mode.
    # -----------------------------------------------------------------------
    # Keepalive is always active in interactive mode.
    # In single-task mode it is off by default (interval=0 disables it).
    keepalive: Optional[ScreenKeepalive] = None
    interval = args.keepalive_interval if args.keepalive_interval > 0 else 30.0
    if args.interactive or args.keepalive_interval > 0:
        keepalive = ScreenKeepalive(
            wda_url=args.wda_url,
            session_id=session_id,
            interval=interval,
            verbose=False,
        )
        keepalive.start()
        print(f"[PhoneClaw] Screen keepalive active (interval: {interval}s)")

    # -----------------------------------------------------------------------
    # 5. Run
    # -----------------------------------------------------------------------
    try:
        if args.learn:
            # Learning / demonstration mode — no planner/evaluator needed
            learn_experience = ExperienceLog(
                log_path=getattr(args, "experience_path", None)
            )
            print(learn_experience.summary_banner())
            _run_learn_mode(
                args=args,
                wda_url=args.wda_url,
                session_id=session_id,
                exec_agent=exec_agent,
                experience=learn_experience,
            )

        elif args.interactive:
            # Memory is managed inside _run_interactive_loop (one instance per session)
            _run_interactive_loop(
                args=args,
                controller=controller,
                executor=executor,
                exec_agent=exec_agent,
                eval_agent=eval_agent,
                planner=planner,
                evaluator=evaluator,
            )
        else:
            # Single-task mode: optionally create memory and experience instances
            single_memory: Optional[UserMemory] = None
            if not args.no_memory:
                single_memory = UserMemory(profile_path=args.memory_path)
                single_memory.start_session()
                print(single_memory.session_banner())

            single_experience: Optional[ExperienceLog] = None
            if not args.no_experience:
                single_experience = ExperienceLog(log_path=args.experience_path)
                print(single_experience.summary_banner())

            _run_single_task(
                task_instruction=args.task,
                args=args,
                controller=controller,
                executor=executor,
                exec_agent=exec_agent,
                eval_agent=eval_agent,
                planner=planner,
                evaluator=evaluator,
                task_dir_override=args.task_dir,
                resume=args.resume,
                memory=single_memory,
                experience=single_experience,
            )
    finally:
        if keepalive:
            keepalive.stop()


if __name__ == "__main__":
    main()
