"""
Warning delivery: on-screen text message shown when a violation fires,
mirroring what the exam portal should display. Per-status messages, e.g.
"Please keep your eyes on the screen." vs. "Please remain quiet and focus
on your exam." rather than one generic warning.
"""

MESSAGES = {
    "looking-away": "Please look at the screen.",
    "eyes-averted": "Please keep your eyes on the screen.",
    "talking": "Please remain quiet and focus on your exam.",
    "no-face-detected": "Please stay in front of the camera.",
    "multiple-faces": "Only the exam candidate should be visible on camera.",
    "too-close": "You are too close to the camera. Please sit properly.",
    "too-far": "Please sit closer and stay within the camera frame.",
}
DEFAULT_MESSAGE = "Please sit properly and look at the screen."


class Alerter:
    def message_for(self, status: str) -> str:
        return MESSAGES.get(status, DEFAULT_MESSAGE)

    def warn(self, status: str):
        message = self.message_for(status)
        print(f"[ALERT] {message}")
        return message
