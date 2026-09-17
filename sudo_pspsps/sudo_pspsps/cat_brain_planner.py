import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import time
import re
import requests
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIG  — set to laptop A's ethernet IP
# ─────────────────────────────────────────────
INFERENCE_SERVER = "http://10.42.0.2:8000"

# SYSTEM_PROMPT removed — it now lives in model_server.py (DEFAULT_SYSTEM_PROMPT).
# Keeping it here and re-transmitting it on every request was the main driver of
# the MPS OOM crash (larger tokenisation → larger intermediate buffers → 17 GB
# Metal allocation failure).  The server falls back to its built-in prompt when
# system_prompt is omitted from the request body.
#
# The intervention-gate call below still sends its own one-off system prompt
# because it needs a different JSON schema ({should_intervene, reason}).

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def parse_llm_json(raw: str) -> dict:
    """Robustly extract JSON from LLM output."""
    raw = re.sub(r"```(?:json)?", "", raw).strip("`").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {
        "action": "CHAT",
        "reasoning": "parse_error",
        "message_to_user": "Sorry, I got a bit confused. Could you repeat that?"
    }


def minutes_until(event_time_str: str) -> float | None:
    """Return minutes until an event given 'HH:MM' string, or None if unparseable."""
    try:
        now = datetime.now()
        t = datetime.strptime(event_time_str, "%H:%M").replace(
            year=now.year, month=now.month, day=now.day
        )
        delta = (t - now).total_seconds() / 60.0
        return delta
    except Exception:
        return None


def build_mood_context(mood: dict) -> str:
    """Build a rich mood context string from VAD + visual rationale."""
    v, a, d = mood.get("vad", [5.0, 5.0, 5.0])
    rationale = mood.get("rationale", "No visual context available.")
    return (
        f"Visual observation from camera: '{rationale}'. "
        f"Emotion scores — valence: {v:.1f}/10, arousal: {a:.1f}/10, dominance: {d:.1f}/10. "
        "The visual observation takes priority if it conflicts with the scores."
    )


# ─────────────────────────────────────────────
#  NODE
# ─────────────────────────────────────────────
class CatBrainPlanner(Node):
    """
    Sudo – the Cat Brain Planner.

    Features
    --------
    1. Idle chit-chat that references the user's emotional state + visual context.
    2. Multi-turn schedule assignment (name → time → date loop until complete).
    3. Proactive reminders when a meeting is ~5 min away.
    4. Mood-aware suggestions: postpone task → play music (max 2), then 10-min back-off.
    """

    REMINDER_THRESHOLD_MIN      = 5.0
    REMINDER_CHECK_INTERVAL_SEC = 30.0
    BACKOFF_DURATION_SEC        = 10 * 60  # 10 minutes
    MAX_SUGGESTIONS             = 2        # postpone + music, then back off
    INTERVENE_VALENCE_GATE      = 4.5      # cheap pre-filter before LLM intervention check

    def __init__(self):
        super().__init__("cat_brain_planner")

        # ── Verify server reachable ───────────────────────────────────────
        self.get_logger().info(f"Checking inference server at {INFERENCE_SERVER}...")
        try:
            r = requests.get(f"{INFERENCE_SERVER}/health", timeout=5.0)
            if r.status_code == 200:
                self.get_logger().info("Inference server reachable ✓")
            else:
                self.get_logger().warn(f"Server responded with {r.status_code}")
        except Exception as e:
            self.get_logger().warn(f"Could not reach server: {e} — will retry on each call")

        # ── Emotional State ───────────────────────────────────────────────
        self.latest_mood: dict = {"vad": [5.0, 5.0, 5.0], "rationale": "Neutral"}

        # ── Schedule Store ────────────────────────────────────────────────
        self.schedule: list[dict] = []

        # ── Active Task-Creation Session ──────────────────────────────────
        self.active_task_draft: dict | None = None

        # ── Reminder Tracking ─────────────────────────────────────────────
        self.reminded_tasks: set = set()

        # ── Proactive Suggestion State ────────────────────────────────────
        self.last_suggestion_time: float = 0.0
        self.suggestion_count: int = 0
        self.waiting_for_suggestion_response: bool = False

        # ── ROS2 Pub/Sub ──────────────────────────────────────────────────
        self.create_subscription(String, "/cat/emotional_state", self.emotion_callback, 10)
        self.create_subscription(String, "/cat/stt_input",       self.stt_callback,     10)
        self.action_pub = self.create_publisher(String, "/cat/robot_actions", 10)

        # ── Reminder Timer ────────────────────────────────────────────────
        self.create_timer(self.REMINDER_CHECK_INTERVAL_SEC, self.reminder_tick)

        self.get_logger().info("CatBrainPlanner node started ✓")

    # ─────────────────────────────────────────────────────────────────────
    #  LLM WRAPPER  — now just an HTTP call
    # ─────────────────────────────────────────────────────────────────────
    def run_llm(self, user_prompt: str, system_prompt: str | None = None) -> dict:
        """Call the inference server.

        system_prompt is optional — omit it to use the server's built-in
        DEFAULT_SYSTEM_PROMPT (the normal Sudo persona).  Pass an explicit
        value only when you need a different response schema, e.g. the
        intervention-gate check that returns {should_intervene, reason}.
        """
        payload: dict = {"user_prompt": user_prompt}
        if system_prompt is not None:
            payload["system_prompt"] = system_prompt

        try:
            resp = requests.post(
                f"{INFERENCE_SERVER}/infer",
                json=payload,
                timeout=30.0,
            )
            resp.raise_for_status()
            raw = resp.json()["result"]
            return parse_llm_json(raw)
        except requests.Timeout:
            self.get_logger().error("Inference server timed out")
        except Exception as e:
            self.get_logger().error(f"Inference server error: {e}")
        return {
            "action": "CHAT",
            "reasoning": "server_error",
            "message_to_user": "Sorry, I'm having trouble thinking right now — give me a moment."
        }

    def publish(self, response: dict):
        self.action_pub.publish(String(data=json.dumps(response)))
        self.get_logger().info(f"→ {response['action']}: {response['message_to_user']}")

    # ─────────────────────────────────────────────────────────────────────
    #  INTERVENTION GATE
    # ─────────────────────────────────────────────────────────────────────
    def _should_intervene(self) -> bool:
        mood_ctx = build_mood_context(self.latest_mood)
        prompt = (
            f"{mood_ctx} "
            "Should the robot assistant proactively check in on the user right now? "
            "Reply ONLY with valid JSON: {\"should_intervene\": true/false, \"reason\": \"...\"}"
        )
        # Pass a lightweight one-off system prompt so the server doesn't apply
        # the full Sudo persona (which expects a different JSON schema).
        gate_system = (
            "You are a concise decision engine. "
            "Always reply with valid JSON only: "
            '{"should_intervene": true/false, "reason": "<short reason>"}'
        )
        result = self.run_llm(prompt, system_prompt=gate_system)
        return bool(result.get("should_intervene", False))

    # ─────────────────────────────────────────────────────────────────────
    #  FEATURE 1 + 4 – EMOTION CALLBACK
    # ─────────────────────────────────────────────────────────────────────
    def emotion_callback(self, msg: String):
        try:
            self.latest_mood = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if self.active_task_draft is not None:
            return

        valence = self.latest_mood["vad"][0]
        now = time.time()
        back_off_expired = (now - self.last_suggestion_time) >= self.BACKOFF_DURATION_SEC

        if back_off_expired:
            self.suggestion_count = 0

        if self.suggestion_count >= self.MAX_SUGGESTIONS or self.waiting_for_suggestion_response:
            return

        if valence < self.INTERVENE_VALENCE_GATE and self._should_intervene():
            self.last_suggestion_time = now
            self.suggestion_count += 1
            self.waiting_for_suggestion_response = True
            self._make_mood_suggestion()

    # ─────────────────────────────────────────────────────────────────────
    #  MOOD SUGGESTION  (Feature 4)
    # ─────────────────────────────────────────────────────────────────────
    def _make_mood_suggestion(self):
        upcoming = self._next_upcoming_task()
        mood_ctx = build_mood_context(self.latest_mood)

        if self.suggestion_count == 1:
            if upcoming:
                prompt = (
                    f"{mood_ctx} "
                    f"Upcoming task: {upcoming}. "
                    "Based on what the camera sees and the emotion scores, "
                    "gently suggest postponing the task so the user can rest. "
                    "Action must be POSTPONE_TASK."
                )
            else:
                prompt = (
                    f"{mood_ctx} "
                    "No upcoming tasks. "
                    "Based on what the camera sees and the emotion scores, "
                    "gently suggest the user takes a break. Action must be SUGGEST_REST."
                )
        else:
            prompt = (
                f"{mood_ctx} "
                "The user declined the previous suggestion. "
                "Based on what the camera sees and the emotion scores, "
                "gently suggest playing some relaxing music. Action must be PLAY_MUSIC."
            )

        response = self.run_llm(prompt)
        self.publish(response)

    # ─────────────────────────────────────────────────────────────────────
    #  FEATURE 3 – REMINDER TICK
    # ─────────────────────────────────────────────────────────────────────
    def reminder_tick(self):
        for task in self.schedule:
            task_id = f"{task.get('person', '')}_{task.get('time', '')}"
            if task_id in self.reminded_tasks:
                continue
            mins = minutes_until(task.get("time", ""))
            if mins is not None and 0 <= mins <= self.REMINDER_THRESHOLD_MIN:
                self.reminded_tasks.add(task_id)
                mood_ctx = build_mood_context(self.latest_mood)
                prompt = (
                    f"Task coming up in {mins:.0f} minute(s): {task}. "
                    f"{mood_ctx} "
                    "Remind the user warmly and ask if they are ready. "
                    "If the visual context suggests they are struggling, be extra gentle. "
                    "Action must be SEND_REMINDER."
                )
                response = self.run_llm(prompt)
                self.publish(response)

    # ─────────────────────────────────────────────────────────────────────
    #  FEATURE 2 – STT CALLBACK
    # ─────────────────────────────────────────────────────────────────────
    def stt_callback(self, msg: String):
        user_input = msg.data.strip()
        if not user_input:
            return

        self.get_logger().info(f"← STT: {user_input}")

        if self.waiting_for_suggestion_response:
            self._handle_suggestion_reply(user_input)
            return

        if self.active_task_draft is not None:
            self._handle_task_draft(user_input)
            return

        if self._looks_like_schedule_request(user_input):
            self._start_task_draft(user_input)
            return

        self._handle_chit_chat(user_input)

    # ─────────────────────────────────────────────────────────────────────
    #  SUGGESTION REPLY HANDLER
    # ─────────────────────────────────────────────────────────────────────
    def _handle_suggestion_reply(self, user_input: str):
        accepted = any(w in user_input.lower() for w in ["yes", "sure", "ok", "okay", "please", "yeah", "yep"])
        declined = any(w in user_input.lower() for w in ["no", "nope", "not", "don't", "dont", "nah"])

        if accepted:
            self.waiting_for_suggestion_response = False
            mood_ctx = build_mood_context(self.latest_mood)
            prompt = (
                f"User accepted the suggestion. {mood_ctx} "
                "Confirm warmly and take action. Use action PLAY_MUSIC or POSTPONE_TASK as appropriate."
            )
            response = self.run_llm(prompt)
            self.publish(response)

        elif declined:
            self.waiting_for_suggestion_response = False
            if self.suggestion_count < self.MAX_SUGGESTIONS:
                self.suggestion_count += 1
                self.waiting_for_suggestion_response = True
                self._make_mood_suggestion()
            else:
                self.get_logger().info("Max suggestions reached. Entering 10-min cool-down.")
                response = {
                    "action": "CHAT",
                    "reasoning": "max suggestions reached",
                    "message_to_user": "Alright, I'll leave you to it! I'm here if you need anything."
                }
                self.publish(response)

        else:
            response = {
                "action": "CHAT",
                "reasoning": "ambiguous reply to suggestion",
                "message_to_user": "Sorry, I didn't quite catch that — would you like me to go ahead, or should I leave you alone for a bit?"
            }
            self.publish(response)

    # ─────────────────────────────────────────────────────────────────────
    #  TASK DRAFT HELPERS  (Feature 2)
    # ─────────────────────────────────────────────────────────────────────
    def _looks_like_schedule_request(self, text: str) -> bool:
        keywords = ["remind", "schedule", "meeting", "appointment", "add", "log", "set a", "book"]
        tl = text.lower()
        return any(k in tl for k in keywords)

    def _start_task_draft(self, user_input: str):
        self.active_task_draft = {"task": None, "person": None, "time": None, "date": None}
        prompt = (
            f"User said: '{user_input}'. "
            "Extract any schedule fields already mentioned (task type, person name, time HH:MM, date YYYY-MM-DD). "
            "Reply as JSON with keys: task, person, time, date — use null for missing fields. "
            "Also set message_to_user to ask for the FIRST missing field. "
            "action must be ASK_TASK_INFO."
        )
        response = self.run_llm(prompt)
        for field in ("task", "person", "time", "date"):
            val = response.get(field)
            if val and val != "null":
                self.active_task_draft[field] = val
        self.publish(response)

    def _handle_task_draft(self, user_input: str):
        missing = [k for k, v in self.active_task_draft.items() if v is None]

        if not missing:
            self._confirm_task()
            return

        prompt = (
            f"Current task draft: {self.active_task_draft}. "
            f"Missing fields: {missing}. "
            f"User just said: '{user_input}'. "
            "Update the draft with any new info found. "
            "If all fields now known, action=CONFIRM_TASK and summarise. "
            "Otherwise action=ASK_TASK_INFO and ask for the next missing field. "
            "Reply as JSON with keys: action, reasoning, message_to_user, "
            "and optional updated fields (task, person, time, date)."
        )
        response = self.run_llm(prompt)

        for field in ("task", "person", "time", "date"):
            val = response.get(field)
            if val and val not in (None, "null"):
                self.active_task_draft[field] = val

        self.publish(response)

        if response.get("action") == "CONFIRM_TASK":
            self._confirm_task()

    def _confirm_task(self):
        entry = dict(self.active_task_draft)
        self.schedule.append(entry)
        self.get_logger().info(f"Task added: {entry}")
        self.active_task_draft = None

    # ─────────────────────────────────────────────────────────────────────
    #  CHIT-CHAT HANDLER  (Feature 1)
    # ─────────────────────────────────────────────────────────────────────
    def _handle_chit_chat(self, user_input: str):
        mood_ctx = build_mood_context(self.latest_mood)
        prompt = (
            f"User said: '{user_input}'. "
            f"{mood_ctx} "
            f"Upcoming schedule: {self.schedule}. "
            "Respond naturally as Sudo, letting the visual observation and emotion scores "
            "shape your tone — e.g. if the camera sees the user crying, be gentle and warm; "
            "if they look relaxed, feel free to be upbeat. "
            "Action must be CHAT."
        )
        response = self.run_llm(prompt)
        self.publish(response)

    # ─────────────────────────────────────────────────────────────────────
    #  UTILS
    # ─────────────────────────────────────────────────────────────────────
    def _next_upcoming_task(self) -> dict | None:
        best = None
        best_mins = float("inf")
        for task in self.schedule:
            mins = minutes_until(task.get("time", ""))
            if mins is not None and mins > 0 and mins < best_mins:
                best_mins = mins
                best = task
        return best


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = CatBrainPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()