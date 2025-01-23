import signal
import sys
from time import time

class Break:
    """Manages conditional breakpoints and handles user interrupts."""
    _enabled = False
    _last_interrupt = None
    _interrupt_grace_period = 1  # Seconds to detect double Ctrl+C

    @classmethod
    def toggle_or_exit(cls, sig, frame):
        """Toggle breakpoint mode or exit on consecutive interrupts."""
        current_time = time()
        if cls._last_interrupt and (current_time - cls._last_interrupt < cls._interrupt_grace_period):
            print("\nExiting program.")
            sys.exit(0)
        cls._last_interrupt = current_time
        cls._enabled = not cls._enabled
        print(f"Breakpoint mode {'enabled' if cls._enabled else 'disabled'}.")

    @classmethod
    def point(cls):
        """Activate a manual breakpoint if enabled."""
        if cls._enabled:
            breakpoint()

    @classmethod
    def start(cls):
        """Attach signal handlers to manage breakpoint toggling."""
        signal.signal(signal.SIGINT, cls.toggle_or_exit)
        print("Press Ctrl+C to toggle breakpoint mode.")
        print("Press Ctrl+C twice quickly to exit.")