"""Ralph Loop orchestrator for PhoneClaw.

Implements the core EXECUTE → EVALUATE → FIX → REPEAT cycle:

  Outer loop: iterate over subtasks (advance on PASS, retry on FAIL)
  Inner loop: execute one action, evaluate, fix if needed

Supports:
  - Filesystem-based state persistence (resume after interruption)
  - Per-subtask fix retry limit
  - Global round count cap
  - Structured logging via PhoneClawRecorder
"""

import re
import sys
import time
from pathlib import Path
from typing import Optional

# Allow running from the Android-Lab root
sys.path.insert(0, str(Path(__file__).parent.parent))

from PhoneClaw.state import TaskState, StateManager, SubTask
from PhoneClaw.recorder import PhoneClawRecorder
from PhoneClaw.prompts import (
    EXECUTOR_SYSTEM_PROMPT,
    EXECUTOR_FIX_CONTEXT_TEMPLATE,
    FINAL_ANSWER_SYSTEM_PROMPT,
    FINAL_ANSWER_USER_TEMPLATE,
)

# Android-Lab utility for extracting code snippets from VLM responses
try:
    from evaluation.definition import get_code_snippet_cot_v3
except ImportError:
    def get_code_snippet_cot_v3(response: str) -> Optional[str]:
        """Fallback: extract text between <CALLED_FUNCTION>…</CALLED_FUNCTION>."""
        match = re.search(r'<CALLED_FUNCTION>\s*(.*?)\s*</CALLED_FUNCTION>', response, re.DOTALL)
        return match.group(1).strip() if match else None


class RalphLoop:
    """
    Ralph Loop controller.

    Usage::

        loop = RalphLoop(
            controller=controller,
            executor=executor,
            agent=agent,
            evaluator=evaluator,
            recorder=recorder,
            state_manager=state_manager,
            max_rounds=100,
            max_fix_retries=3,
            request_interval=2.0,
        )
        loop.run(state)
    """

    def __init__(
        self,
        controller,
        executor,
        agent,
        evaluator,
        recorder: PhoneClawRecorder,
        state_manager: StateManager,
        max_rounds: int = 100,
        max_fix_retries: int = 3,
        request_interval: float = 2.0,
        skip_failed_subtasks: bool = True,
        experience=None,
    ):
        """
        Args:
            controller: IOSController for WDA device control.
            executor: IOSExecutor for action dispatch.
            agent: VLM agent with act() and prompt_to_message_visual() methods.
            evaluator: SubTaskEvaluator instance.
            recorder: PhoneClawRecorder for trace logging.
            state_manager: StateManager for filesystem persistence.
            max_rounds: Global cap on total action rounds across all subtasks.
            max_fix_retries: Max fix attempts per subtask before giving up and advancing.
            request_interval: Seconds to sleep between action rounds.
            skip_failed_subtasks: If True, advance to next subtask after max_fix_retries.
                                   If False, abort the entire task.
            experience: Optional ExperienceLog instance.  When provided, relevant
                        lessons are injected into every Executor prompt, and new
                        lessons are extracted after the task completes.
        """
        self.controller = controller
        self.executor = executor
        self.agent = agent
        self.evaluator = evaluator
        self.recorder = recorder
        self.state_manager = state_manager
        self.max_rounds = max_rounds
        self.max_fix_retries = max_fix_retries
        self.request_interval = request_interval
        self.skip_failed_subtasks = skip_failed_subtasks
        self.experience = experience

        # Set by run(); callers (e.g. run_phoneclaw) can read this after run() returns
        self.last_final_answer: Optional[str] = None

        # Tracks which app is currently in the foreground (inferred from launch() calls)
        self._current_app: Optional[str] = None
        # Per-subtask execution log built during run(); used for experience extraction
        self._subtask_logs: list[dict] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, state: TaskState) -> TaskState:
        """
        Execute the Ralph Loop until all subtasks pass or limits are reached.

        Args:
            state: Initial (or resumed) TaskState.

        Returns:
            Final TaskState after the loop completes.
        """
        # Reset per-task state
        self._current_app = None
        self._subtask_logs = []

        print(f"\n{'='*60}")
        print(f"[RalphLoop] Starting task: {state.task_instruction}")
        print(f"[RalphLoop] Subtasks: {len(state.subtasks)}, Max rounds: {self.max_rounds}")
        print(f"{'='*60}\n")

        while not state.is_complete and state.round_count < self.max_rounds:
            subtask = state.current_subtask
            if subtask is None:
                break

            print(f"\n[RalphLoop] --- Subtask #{subtask.id}: {subtask.instruction} ---")
            print(f"[RalphLoop]     Criteria: {subtask.success_criteria}")

            # Update recorder with current subtask context
            self.recorder.set_current_subtask(
                idx=state.current_subtask_idx,
                instruction=subtask.instruction,
                criteria=subtask.success_criteria,
            )

            # Reset executor's finish flag for this subtask
            self.executor.reset_finish()

            # --- EXECUTE phase ---
            fix_hint: Optional[str] = None
            advanced = self._execute_subtask(state, subtask, fix_hint)

            if advanced:
                # Successfully advanced: persist state and continue outer loop
                self.state_manager.save(state)
                continue

            # If we reach here the subtask either ran out of fix retries or was skipped
            self.state_manager.save(state)

            if state.status == "failed":
                break

        # Mark task as completed if all subtasks done
        if state.is_complete and state.status == "running":
            state.status = "completed"
        elif state.round_count >= self.max_rounds and not state.is_complete:
            state.status = "failed"
            print(f"\n[RalphLoop] Max rounds ({self.max_rounds}) reached. Task incomplete.")

        # --- FINAL ANSWER: extract the answer to the user's question ---
        final_answer: Optional[str] = None
        if state.status == "completed":
            final_answer = self._generate_final_answer(state)

        # Expose for callers (e.g. run_phoneclaw memory recording)
        self.last_final_answer = final_answer

        # --- EXPERIENCE: extract lessons from this task's trace ---
        if self.experience is not None and self._subtask_logs:
            print("\n[Experience] Extracting lessons from task trace...")
            self.experience.extract_and_record(
                task=state.task_instruction,
                subtask_logs=self._subtask_logs,
                final_answer=final_answer,
                agent=self.agent,
            )

        self.state_manager.save(state)
        self.recorder.log_task_complete(
            all_passed=state.status == "completed",
            summary=state.summary(),
            final_answer=final_answer,
        )

        print(f"\n{'='*60}")
        print(f"[RalphLoop] Task finished. Status: {state.status}")
        print(state.summary())
        if final_answer:
            print(f"\n{'='*60}")
            print("[PhoneClaw] ANSWER")
            print(f"{'='*60}")
            print(final_answer)
            print(f"{'='*60}\n")
        print(f"{'='*60}\n")

        return state

    # ------------------------------------------------------------------
    # Final answer extraction (runs once after all subtasks complete)
    # ------------------------------------------------------------------

    def _generate_final_answer(self, state: TaskState) -> Optional[str]:
        """
        After all subtasks pass, take a fresh screenshot and ask the VLM to
        directly answer the user's original question based on what is on screen.

        Also uses any finish() message stored by the executor as a fallback.

        Returns:
            The answer string, or None if extraction failed.
        """
        # If the executor already recorded an explicit finish() answer, use it
        # as a first-pass hint (we still run the VLM for a clean natural-language answer)
        finish_hint = getattr(self.executor, "finish_message", None)

        try:
            # Take a fresh screenshot for the final answer step
            self.executor.update_screenshot(prefix="final_answer")
            screenshot = self.executor.current_screenshot_path

            user_content = FINAL_ANSWER_USER_TEMPLATE.format(
                task_instruction=state.task_instruction,
            )

            # If the executor captured information via finish(), include it as context
            if finish_hint:
                user_content += (
                    f"\n\nNote: the agent's last action reported: \"{finish_hint}\""
                )

            system_msg = {"role": "system", "content": FINAL_ANSWER_SYSTEM_PROMPT}
            user_messages = self.agent.prompt_to_message_visual(user_content, screenshot)
            messages = [system_msg, *user_messages]

            print("\n[RalphLoop] Generating final answer...")
            answer = self.agent.act(messages)
            return answer.strip() if answer else None

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[RalphLoop] Warning: could not generate final answer: {e}")
            return finish_hint  # fall back to executor's finish() message

    # ------------------------------------------------------------------
    # Subtask execution loop
    # ------------------------------------------------------------------

    def _execute_subtask(
        self,
        state: TaskState,
        subtask: SubTask,
        initial_fix_hint: Optional[str],
    ) -> bool:
        """
        Inner EXECUTE → EVALUATE → FIX → REPEAT loop for a single subtask.

        Returns:
            True if the subtask passed and state was advanced.
            False if max retries exceeded or task aborted.
        """
        fix_hint = initial_fix_hint

        # Accumulate (action, eval_reason) pairs for every failed attempt so the
        # VLM can see exactly what it already tried and avoid repeating itself.
        failed_actions: list[str] = []

        # Detect current app from launch() call in the subtask instruction
        app_match = re.search(r'launch\("([^"]+)"\)', subtask.instruction)
        if app_match:
            self._current_app = app_match.group(1)

        # Per-subtask execution log entry for experience extraction
        subtask_entry: dict = {
            "instruction": subtask.instruction,
            "app": self._current_app,
            "passed": False,
            "actions": [],  # list of {action, passed, reason}
        }

        # Each subtask gets its own action rounds, bounded by global max_rounds
        while state.round_count < self.max_rounds:
            state.round_count += 1
            print(f"\n[RalphLoop] Round {state.round_count} | Subtask #{subtask.id} | "
                  f"Fix attempt {subtask.fix_retries}/{self.max_fix_retries}")

            # --- EXECUTE: one action step ---
            rsp, exe_res, code_snippet, prompt_his = self._run_action_step(
                state, subtask, fix_hint, failed_actions
            )

            # Brief pause to let the UI settle
            time.sleep(self.request_interval)

            # --- EVALUATE ---
            # Take a fresh screenshot after the action for evaluation
            self.executor.update_screenshot(prefix=f"eval_{state.round_count}")
            eval_screenshot = (
                self.executor.current_screenshot_path
                or self.recorder.current_screenshot_path
            )

            eval_result = self.evaluator.evaluate(
                screenshot_path=eval_screenshot,
                success_criteria=subtask.success_criteria,
            )

            print(f"[Evaluator] Result: {'PASS' if eval_result.passed else 'FAIL'} — {eval_result.reason}")

            # Record this action in the subtask log
            subtask_entry["actions"].append({
                "action": code_snippet or "(no action extracted)",
                "passed": eval_result.passed or self.executor.is_finish,
                "reason": eval_result.reason[:120],
            })

            # Log the step with eval result
            self.recorder.update_after_cot(
                exe_res=exe_res,
                response=rsp,
                prompt_his=prompt_his,
                code_snippet=code_snippet,
                eval_result=eval_result.to_dict(),
                fix_attempt=subtask.fix_retries,
            )
            self.recorder.turn_number += 1

            # --- PASS: advance to next subtask ---
            if eval_result.passed or self.executor.is_finish:
                reason = eval_result.reason if eval_result.passed else "Agent called finish()"
                subtask_entry["passed"] = True
                self._subtask_logs.append(subtask_entry)
                state.mark_current_passed(reason)
                self.recorder.log_subtask_result(
                    subtask_idx=state.current_subtask_idx,
                    subtask={"instruction": subtask.instruction, "success_criteria": subtask.success_criteria},
                    passed=True,
                    reason=reason,
                )
                state.advance()
                print(f"[RalphLoop] Subtask #{subtask.id} PASSED. Advancing.")
                return True

            # --- FAIL: record what failed, check retry budget ---

            # Build a concise failure entry: "action → short reason"
            action_label = code_snippet or "(no action extracted)"
            reason_short = eval_result.reason[:100]
            repeat_warning = ""

            # Detect repeated identical action — make it explicit in the log
            if failed_actions and action_label != "(no action extracted)":
                prev_actions = [e.split(" →")[0].strip() for e in failed_actions]
                repeat_count = prev_actions.count(action_label)
                if repeat_count >= 1:
                    repeat_warning = f"  ⚠ REPEATED {repeat_count + 1}×"
                    print(f"[RalphLoop] Warning: identical action '{action_label}' "
                          f"has been tried {repeat_count + 1} time(s) and keeps failing.")

            failed_actions.append(f"  {action_label} → \"{reason_short}\"{repeat_warning}")

            subtask.fix_retries += 1

            if subtask.fix_retries > self.max_fix_retries:
                subtask_entry["passed"] = False
                self._subtask_logs.append(subtask_entry)
                state.mark_current_failed(eval_result.reason)
                self.recorder.log_subtask_result(
                    subtask_idx=state.current_subtask_idx,
                    subtask={"instruction": subtask.instruction, "success_criteria": subtask.success_criteria},
                    passed=False,
                    reason=eval_result.reason,
                )
                print(f"[RalphLoop] Subtask #{subtask.id} FAILED after {self.max_fix_retries} retries.")

                if self.skip_failed_subtasks:
                    print(f"[RalphLoop] Skipping to next subtask.")
                    state.advance()
                    return False
                else:
                    print(f"[RalphLoop] Aborting task.")
                    state.status = "failed"
                    return False

            # --- FIX: pass latest evaluator reason forward ---
            fix_hint = eval_result.reason
            print(f"[RalphLoop] FIX attempt {subtask.fix_retries}/{self.max_fix_retries}. "
                  f"Hint: {fix_hint[:80]}...")

        # Global round cap reached inside inner loop
        return False

    # ------------------------------------------------------------------
    # Single action step
    # ------------------------------------------------------------------

    def _run_action_step(
        self,
        state: TaskState,
        subtask: SubTask,
        fix_hint: Optional[str],
        failed_actions: Optional[list] = None,
    ):
        """
        Capture screenshot, call VLM, execute action, update recorder.

        Args:
            failed_actions: List of strings describing previously failed actions
                            for this subtask (injected into the fix context so the
                            VLM knows exactly what NOT to repeat).

        Returns:
            (response, exe_res, code_snippet, prompt_his)
        """
        # Capture screenshot + XML (no labeled overlay — agent uses raw coordinates)
        self.recorder.update_before(
            controller=self.controller,
            need_screenshot=True,
            need_labeled=False,
        )

        image_path = self.recorder.current_screenshot_path

        # Build executor system prompt with current subtask context
        fix_context = ""
        if fix_hint and subtask.fix_retries > 0:
            summary = (
                "\n".join(failed_actions)
                if failed_actions
                else "  (none recorded)"
            )
            fix_context = EXECUTOR_FIX_CONTEXT_TEMPLATE.format(
                fix_attempt=subtask.fix_retries,
                fail_reason=fix_hint,
                success_criteria=subtask.success_criteria,
                failed_actions_summary=summary,
            )

        # Inject relevant past-execution hints from the experience log
        experience_notes = ""
        if self.experience is not None:
            experience_notes = self.experience.get_hints_for(
                app_name=self._current_app,
                subtask_instruction=subtask.instruction,
            )

        system_content = (
            EXECUTOR_SYSTEM_PROMPT.format(
                subtask_instruction=subtask.instruction,
                overall_task=state.task_instruction,
            )
            + fix_context
            + experience_notes
        )

        system_msg = {"role": "system", "content": system_content}

        # Build user message with history + current screenshot
        history_tail = self.recorder.history[-4:] if self.recorder.history else []
        history_text = "\n".join(history_tail) if history_tail else "[]"
        user_text = (
            f"Current subtask: {subtask.instruction}\n"
            f"History:\n{history_text}\n"
            f"Current screen: <image>"
        )

        try:
            user_messages = self.agent.prompt_to_message_visual(user_text, image_path)
            messages = [system_msg, *user_messages]
            rsp = self.agent.act(messages)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rsp = f"Error calling agent: {e}"

        # Extract and execute the code snippet
        code_snippet = get_code_snippet_cot_v3(rsp)

        if code_snippet:
            try:
                exe_res = self.executor(code_snippet)
            except Exception as e:
                print(f"[RalphLoop] Error executing code snippet: {e}")
                exe_res = {"operation": "error", "action": "error", "kwargs": {"error": str(e)}}
        else:
            print("[RalphLoop] Warning: Could not extract code snippet from response.")
            exe_res = {"operation": "skip", "action": "skip", "kwargs": {"reason": "No code snippet"}}

        # Extract state assessment for history
        pattern = r'<STATE_ASSESSMENT>\s*(.*?)\s*</STATE_ASSESSMENT>'
        match = re.search(pattern, rsp, re.DOTALL)
        prompt_his = match.group(1) if match else None

        return rsp, exe_res, code_snippet, prompt_his
