# PhoneClaw — iOS Ralph Loop GUI Agent

An autonomous iOS GUI automation agent built on the **Ralph Loop** methodology:  
**EXECUTE → EVALUATE → FIX → REPEAT**, repeating until every subtask passes its success criteria.

Device control is provided by WebDriverAgent (WDA). Key features include structured LLM-driven planning, per-subtask evaluation with automatic retries, natural-language answer extraction, a persistent interactive daemon mode, and a **two-layer self-learning memory** that makes the agent progressively smarter across sessions.

---

## Architecture

```
run_phoneclaw.py
    │
    ├─ ScreenKeepalive      ← background: keep device screen on
    ├─ UserMemory           ← persistent user profile (facts, habits, history)
    ├─ ExperienceLog        ← persistent execution know-how (what worked / failed)
    ├─ TaskPlanner          ← VLM: task → subtask list with success criteria
    ├─ StateManager         ← filesystem persistence (phoneclaw_state.json)
    │
    └─ RalphLoop (loop.py)
          │
          │  for each subtask:
          │
          ├─ EXPERIENCE HINTS ─ ExperienceLog.get_hints_for(app, subtask)
          │                      injected into Executor system prompt
          │
          ├─ EXECUTE ─── IOSExecutor(code_snippet)
          │                  ├─ VLM outputs normalised relative coords, e.g. tap(0.50, 0.90)
          │                  ├─ Executor converts [0,1] → physical pixels → WDA logical coords
          │                  └─ IOSActionHandler → WDA HTTP → iOS device
          │
          ├─ EVALUATE ── SubTaskEvaluator(screenshot, criteria)
          │                  └─ VLM: screenshot + criteria → {passed, reason}
          │
          ├─ PASS ──────── advance to next subtask, persist state
          │
          └─ FAIL ──────── increment fix_retries
                           retries < max → FIX (inject fail reason + action history)
                                         → EXECUTE again
                           retries ≥ max → mark failed, skip / abort
          │
          └─ FINAL ANSWER ── VLM reads current screen + original question → answer string
          │
          └─ LEARN ────────── ExperienceLog.extract_and_record(trace)
                               UserMemory.extract_insights(task, answer)
```

---

## Self-Learning Memory

PhoneClaw accumulates two complementary memory stores that persist across sessions:

### UserMemory — who the user is

Stored in `PhoneClaw/data/user_profile.json`.

| Category | Examples |
|---|---|
| Profile | Inferred name, location, language |
| App usage | Meituan used 8×, WeChat used 5× |
| Task history | Last 200 tasks with answers and status |
| Insights | "User is located in Hangzhou", "Frequently orders Heytea delivery", "Meituan account ********" |

**How it helps the Planner:** A `## User Profile` block is injected into every Planner prompt, so the agent can make smarter subtask plans (e.g. knowing the user's city improves location-sensitive searches).

**Memory-first retrieval:** Before touching the device, the agent checks whether the profile already contains a confident answer to the question. If yes, the answer is returned immediately with zero device interactions.

```
[PhoneClaw] Task> What is the name of my Meituan account ？
[Memory] Checking profile for cached answer...
[Memory] Answer found in profile — skipping device interaction.
═══════════════════════════════════════
[PhoneClaw] ANSWER  (from memory)
═══════════════════════════════════════
Your account name is ********
```

### ExperienceLog — how to do things

Stored in `PhoneClaw/data/experience_log.json`.

| Lesson type | Example |
|---|---|
| `successful_navigation` | "Meituan orders tab coordinate approx. (0.62, 0.94)" |
| `failed_approach` | "Tapping (0.5, 0.5) on Meituan home screen triggers an ad popup" |
| `ui_knowledge` | "WeChat search bar is at the top, y≈0.07" |
| `timing` | "Meituan order list takes approx. 3 seconds to load" |

**How it helps the Executor:** Before every Executor VLM call, relevant lessons for the current app are injected into the system prompt, so the model avoids previously-failed actions and reuses previously-successful navigation paths.

**Reinforcement:** When a lesson is confirmed by a later task, its `reinforced` counter increments and its `confidence` may be upgraded (`low → medium → high`).

**Semantic deduplication:** New lessons and insights are compared to existing ones using OpenAI-compatible text embeddings (cosine similarity, threshold 0.88). A Jaccard similarity fallback (threshold 0.50) is used when no embedding API key is available.

**Automatic compaction:** When an app accumulates ≥ 20 lessons, a VLM-driven compaction pass merges near-duplicate entries, removes low-value items, and generalises specific coordinates — targeting ≤ 8 high-quality lessons per app.

---

## Complete Interactive-Mode Flow

```
Start session
    │
    ├─ Load UserMemory         (profile + history from data/user_profile.json)
    ├─ Load ExperienceLog      (lessons from data/experience_log.json)
    ├─ Start ScreenKeepalive   (device stays awake)
    │
    │  User types task
    │      │
    │      ▼
    ├─ [1] Memory-first query  ─── answered from profile? ──► return answer, done
    │      │ not found
    │      ▼
    ├─ [2] Plan  (Planner + user context injected from UserMemory)
    │      │
    │      ▼
    ├─ [3] Ralph Loop  (for each subtask)
    │        ├─ inject ExperienceLog hints into Executor prompt
    │        └─ EXECUTE → EVALUATE → FIX → REPEAT
    │      │
    │      ▼
    ├─ [4] Final Answer  (VLM reads screen → natural-language answer)
    │      │
    │      ▼
    ├─ [5] Record task  (UserMemory.record_task)
    ├─ [6] Extract insights  (VLM → new user facts → UserMemory)
    └─ [7] Extract lessons   (VLM → new app lessons → ExperienceLog)
              └─ compact_if_needed()  (auto-compact when lessons ≥ 20)
```

---

## Coordinate System

VLM outputs **normalised relative coordinates** in `[0.0, 1.0]`:

```
(0.0, 0.0) ─────────────── (1.0, 0.0)   top
     │                           │
     │       (0.5, 0.5)          │   centre
     │                           │
(0.0, 1.0) ─────────────── (1.0, 1.0)   bottom
```

| Screen area | x range | y range |
|---|---|---|
| Status bar | any | 0.02 – 0.06 |
| Top navigation | any | 0.06 – 0.12 |
| Centre | ~0.50 | ~0.50 |
| Bottom tab bar | any | 0.90 – 0.96 |

The Executor converts relative coords → physical pixels → WDA logical coords internally. The VLM never needs to know the device resolution.

---

## Directory Structure

```
PhoneClaw/
├── run_phoneclaw.py     # CLI entry point (single-task + interactive daemon)
│
├── loop.py              # Ralph Loop orchestrator (EXECUTE → EVALUATE → FIX)
├── planner.py           # VLM task decomposition → subtask list
├── evaluator.py         # VLM screenshot evaluation → pass/fail
├── state.py             # Filesystem state persistence (phoneclaw_state.json)
├── prompts.py           # All VLM prompt templates
├── agent.py             # OpenRouterAgent (OpenRouter API)
├── keepalive.py         # Screen keepalive (idleTimerDisabled / touch fallback)
│
├── memory.py            # UserMemory: user profile + task history + insights
├── experience.py        # ExperienceLog: app-specific execution lessons
├── embeddings.py        # Semantic deduplication (embedding cosine / Jaccard)
├── learn.py             # DemoRecorder: learning mode (record user demos)
│
├── actions.py           # WDA HTTP action primitives + iOS bundle ID map
├── connection.py        # WDA session management
├── controller.py        # IOSController (Android-Lab compatible interface)
├── executor.py          # IOSExecutor: coord conversion + action dispatch
├── hierarchy.py         # XML page source → IOSElement list
├── labeling.py          # Draw bounding boxes on screenshots
├── screenshot.py        # Screenshot capture via WDA / idevicescreenshot
├── recorder.py          # Per-step JSONL trace logging (PhoneClawRecorder)
│
└── data/                # Persistent data (auto-created on first run)
    ├── user_profile.json
    ├── experience_log.json
    └── demos/           # Learning-mode demo recordings
```

Runtime log directories (created on each task run):

```
phoneclaw_logs/<task_id>/
├── phoneclaw_state.json  # Subtask list, progress, fix counts
├── traces/trace.jsonl    # Per-step trace: screenshots, VLM responses, eval results
├── screenshots/          # Raw screenshots per round
└── xml/                  # iOS page source XML per round
```

---

## Requirements

- **iOS device** with **WebDriverAgent** running
- Python packages: `requests`, `Pillow`, `opencv-python`, `openai`, `backoff`, `lxml`
- Optional: `libimobiledevice` (`idevicescreenshot`) for screenshot fallback

---

## Quick Start

### 1. Start WebDriverAgent on the device

```bash
iproxy 8100 8100
```

### 2. Configure the VLM backend

**Option A — OpenRouter (recommended)**

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

**Option B — Local / self-hosted model**

```bash
export API_BASE="http://localhost:8002/v1"
export MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
```

Or copy `.env.example` to `.env` and edit it.

### 3a. Single-task mode

```bash
cd Android-Lab

python PhoneClaw/run_phoneclaw.py \
    --task "Open Meituan and check what the most recent food delivery order was" \
    --openrouter \
    --model-name "z-ai/glm-4.6v"
```

Example output:

```
[Memory] Checking profile for cached answer...
[Memory] Not in profile — will use device.

[PhoneClaw] Planning subtasks...
[Planner] Decomposed task into 3 subtask(s).
  #1: Call launch("Meituan") to open Meituan
  #2: Navigate to the orders page
  #3: Read the most recent order and call finish() with the answer

... (execution) ...

[Experience] Extracting lessons from task trace...
[Experience] new: [Meituan] Meituan orders tab coordinate approx. (0.62, 0.94)

════════════════════════════════════════════════════
[PhoneClaw] ANSWER
════════════════════════════════════════════════════
The most recent order is from Heytea (Hangzhou Wanxiang Center):
Hot Roasted Brown Sugar Boba Milk Tea ¥21.9, placed on 2026-02-19 12:21.
════════════════════════════════════════════════════
```

Subsequent identical questions are answered from memory with no device interaction:

```
[Memory] Answer found in profile — skipping device interaction.
```

### 3b. Interactive / daemon mode

Connect once, accept tasks from stdin indefinitely. The device screen stays on automatically.

```bash
python PhoneClaw/run_phoneclaw.py \
    --interactive \
    --openrouter \
    --model-name "z-ai/glm-4.6v"
```

```
════════════════════════════════════════════════════
[PhoneClaw] Interactive mode — device connected.
[Memory] Profile: PhoneClaw/data/user_profile.json
[Memory] Sessions: 4  |  Tasks: 12 completed / 1 failed  |  Insights: 8
[Experience] Log: PhoneClaw/data/experience_log.json  |  Lessons: 23  |  Tasks: 11
[PhoneClaw] Commands: 'memory' — profile  |  'experience' — lessons  |  'quit' — exit
════════════════════════════════════════════════════

[PhoneClaw] Task> Open WeChat and view the 5 most recent conversations
... (executes) ...

[PhoneClaw] Task> What is my Meituan account name?
[Memory] Answer found in profile — skipping device interaction.
Your Meituan account name is Alice123

[PhoneClaw] Task> memory
  ══════════════════════════════════════════════════════════
  User Profile  —  PhoneClaw/data/user_profile.json
  ══════════════════════════════════════════════════════════
  Sessions     : 4
  Tasks total  : 13  (✓ 12  ✗ 1)
  Insights     : 8
  Name         : Alice123
  Location     : Hangzhou

  App usage:
    Meituan              8×  (last: 2026-03-09)
    WeChat               5×  (last: 2026-03-09)

  Insights:
    • User's Meituan account name is Alice123
    • User frequently orders Heytea delivery near Hangzhou Wanxiang Center
    • User typically orders food via Meituan at lunchtime

[PhoneClaw] Task> experience
  ══════════════════════════════════════════════════════════
  Experience Log  —  PhoneClaw/data/experience_log.json
  ══════════════════════════════════════════════════════════
  Lessons: 23  |  Tasks processed: 11

  [Meituan]
    ✓[H×4] View Meituan order history: tap the 'Orders' tab at the bottom, coord approx. (0.62, 0.94)
    ✗[H×2] Tapping (0.5, 0.5) on Meituan home triggers a promotional popup, not the orders page
    ℹ[M×1] Meituan bottom tabs: Home x≈0.12 / Nearby x≈0.38 / Orders x≈0.62 / Mine x≈0.88

  [WeChat]
    ✓[H×3] WeChat chat list is visible immediately after launch; no extra navigation needed
    ℹ[M×2] WeChat search bar is at the top, coord approx. (0.5, 0.07)

[PhoneClaw] Task> compact
[PhoneClaw] Running full experience compaction (may take a minute)...

[PhoneClaw] Task> quit
[PhoneClaw] Goodbye.
```

**Special commands in interactive mode:**

| Command | Aliases | Description |
|---|---|---|
| `memory` | `profile` | Display user profile summary |
| `experience` | `exp`, `lessons` | Display experience log by app |
| `compact` | — | Trigger manual compaction of experience log |
| `quit` | `exit`, `q` | Exit cleanly |

- **Ctrl+C inside a task** — interrupts that task only; ready for the next
- **Ctrl+C at the prompt / `quit`** — exits cleanly
- Each task creates its own log directory under `./phoneclaw_logs/`

### 3c. Learning / demonstration mode

Record your own manual device operations so PhoneClaw can learn from them.

**Prerequisites:** Enable *Settings → Accessibility → Touch → Show Touches* on the device so tap positions can be detected from screenshots.

```bash
python PhoneClaw/run_phoneclaw.py \
    --learn \
    --learn-app "Xiaohongshu" \
    --learn-describe "browsing the discovery feed" \
    --openrouter \
    --model-name "z-ai/glm-4.6v"
```

PhoneClaw captures screenshots at ~8 fps, detects tap positions via OpenCV `HoughCircles` (falls back to pixel-diff centroid), annotates each frame, and then calls the VLM to extract reusable lessons that are added directly to the ExperienceLog.

Annotated frames are saved to `PhoneClaw/data/demos/<app>_<timestamp>/`.

---

## OpenRouter Configuration

`agent.py` provides `OpenRouterAgent`:

- Standard `image_url` data-URI format (compatible with all OpenRouter vision models)
- Required `HTTP-Referer` and `X-Title` headers
- Auto-resizes screenshots to fit model context limits
- Exponential backoff, up to 5 retries on transient API errors

### Separate executor and evaluator models

```bash
python PhoneClaw/run_phoneclaw.py \
    --task "Send a message to Alice on WeChat" \
    --openrouter \
    --model-name "z-ai/glm-4.6v" \
    --eval-model-name "openai/gpt-4o"
```

---

## CLI Reference

### Mode arguments

| Argument | Default | Description |
|---|---|---|
| `--task TEXT` | — | Task description (required in single-task mode) |
| `--interactive` | off | Daemon mode: connect once, accept tasks indefinitely |
| `--learn` | off | Learning mode: record demo and extract lessons |

### Loop / execution arguments

| Argument | Default | Description |
|---|---|---|
| `--wda-url URL` | `$WDA_URL` / `http://localhost:8100` | WebDriverAgent base URL |
| `--max-rounds N` | `100` | Global cap on total action rounds |
| `--max-fix-retries N` | `3` | Max fix attempts per failing subtask |
| `--no-skip-failed` | off | Abort entire task on subtask failure |
| `--request-interval S` | `2.0` | Seconds between action rounds |

### Logging / resume arguments

| Argument | Default | Description |
|---|---|---|
| `--task-dir PATH` | auto-generated | Override log output directory |
| `--resume` | off | Resume from saved state in `--task-dir` |

### Memory arguments

| Argument | Env var | Default | Description |
|---|---|---|---|
| `--memory-path PATH` | `PHONECLAW_MEMORY` | `PhoneClaw/data/user_profile.json` | User profile JSON path |
| `--no-memory` | — | off | Disable memory recording for this run |

### Experience arguments

| Argument | Env var | Default | Description |
|---|---|---|---|
| `--experience-path PATH` | `PHONECLAW_EXPERIENCE` | `PhoneClaw/data/experience_log.json` | Experience log JSON path |
| `--no-experience` | — | off | Disable experience recording / injection |

### Screen keepalive

| Argument | Env var | Default | Description |
|---|---|---|---|
| `--keepalive-interval S` | `KEEPALIVE_INTERVAL` | `30` | Heartbeat / fallback-tap interval (seconds). Always active in `--interactive`. Set `0` to disable. |

### OpenRouter arguments

| Argument | Env var | Default | Description |
|---|---|---|---|
| `--openrouter` | `OPENROUTER_API_KEY` (auto-detect) | off | Use OpenRouter backend |
| `--openrouter-api-key KEY` | `OPENROUTER_API_KEY` | — | API key |
| `--model-name SLUG` | `OPENROUTER_MODEL` | `z-ai/glm-4.6v` | Executor model |
| `--eval-model-name SLUG` | `EVAL_OPENROUTER_MODEL` | same as `--model-name` | Evaluator model |
| `--openrouter-base-url URL` | — | `https://openrouter.ai/api/v1` | API endpoint |
| `--openrouter-site-url URL` | `OPENROUTER_SITE_URL` | — | HTTP-Referer header |
| `--openrouter-app-title STR` | `OPENROUTER_APP_TITLE` | `PhoneClaw` | X-Title header |

### Local VLM arguments

| Argument | Env var | Default | Description |
|---|---|---|---|
| `--api-base URL` | `API_BASE` | `http://localhost:8002/v1` | Executor endpoint |
| `--model-name NAME` | `MODEL_NAME` | `Qwen/Qwen2.5-VL-7B-Instruct` | Executor model |
| `--api-key KEY` | `API_KEY` | `EMPTY` | Executor API key |
| `--agent-type TYPE` | `AGENT_TYPE` | `OpenAIAgent` | `OpenAIAgent` or `QwenVLAgent` |
| `--eval-api-base URL` | `EVAL_API_BASE` | same as `--api-base` | Evaluator endpoint |
| `--eval-model-name NAME` | `EVAL_MODEL_NAME` | same as `--model-name` | Evaluator model |

### Learning-mode arguments

| Argument | Default | Description |
|---|---|---|
| `--learn-app NAME` | — | App name to record (required with `--learn`) |
| `--learn-describe TEXT` | — | Optional description of what you are demonstrating |
| `--learn-duration S` | `60` | Max recording duration in seconds |
| `--learn-poll MS` | `125` | Screenshot polling interval (milliseconds) |
| `--learn-threshold N` | `30` | Pixel-diff threshold for change detection |
| `--learn-dir PATH` | `PhoneClaw/data/demos/` | Override demo output directory |
| `--no-analyse` | off | Record frames only; skip VLM analysis |

---

## Screen Keepalive

`keepalive.py` prevents the device screen from sleeping using a two-tier strategy:

**Primary — `idleTimerDisabled` (no UI interaction)**  
On `start()`, PhoneClaw sends `POST /wda/settings {"settings": {"idleTimerDisabled": true}}` to the WDA session. This disables iOS's auto-lock at the system level for the lifetime of the session, with zero UI side-effects. A lightweight `GET /status` heartbeat thread keeps the WDA HTTP session alive. On `stop()`, `idleTimerDisabled` is reset to `false`.

**Fallback — periodic centre tap**  
Older or custom WDA builds may not support the `idleTimerDisabled` setting. In that case, PhoneClaw falls back to a periodic synthetic tap at screen centre (0.50, 0.50) via WDA W3C pointer actions.

---

## Supported Apps (via `launch()`)

`launch("AppName")` opens apps directly by bundle ID — faster and more reliable than tapping the home screen icon. Both English and Chinese app names are resolved via bundle ID, so `launch("Xiaohongshu")` and its Chinese alias both refer to the same app.

| Category | Supported apps |
|---|---|
| System | Settings, Safari, Calendar, Messages, Mail, Photos, Camera, Clock, Maps, Music, App Store, Notes, Reminders, Weather, Calculator, Contacts, FaceTime, Phone |
| Social | WeChat, QQ, Weibo, Feishu / Lark |
| Shopping | Meituan, Taobao, JD, Pinduoduo, Xiaohongshu |
| Travel | Didi, Ctrip |
| Finance | Alipay |
| Video | Douyin, Bilibili, iQIYI, Youku, Tencent Video |
| Music | NetEase Music, QQ Music |
| Google | Gmail, Google Maps, Google Chrome, YouTube |
| Other | Zhihu, Baidu Maps, Gaode Maps |

---

## Output Format

### Per-step trace entry

```json
{
  "subtask_idx": 1,
  "subtask_instruction": "Navigate to the orders page",
  "subtask_criteria": "Order list is visible with past orders",
  "image": "screenshots/screenshot-2-before.png",
  "response": "<CALLED_FUNCTION>tap(0.62, 0.94)</CALLED_FUNCTION>...",
  "code_snippet": "tap(0.62, 0.94)",
  "eval_result": {"passed": true, "reason": "Order list is visible"},
  "fix_attempt": 0
}
```

### Task completion entry

```json
{
  "type": "task_complete",
  "all_passed": true,
  "summary": "Task: ... Progress: 3/3 passed ...",
  "final_answer": "The most recent order is from Heytea (Hangzhou Wanxiang Center): Hot Roasted Brown Sugar Boba Milk Tea ¥21.9"
}
```
