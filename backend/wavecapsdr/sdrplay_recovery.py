"""SDRplay service recovery module.

Automatically detects and recovers from stuck SDRplay API service.
"""

import logging
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RecoveryStats:
    """Statistics for recovery operations."""

    last_recovery_attempt: Optional[float] = None
    last_recovery_success: Optional[float] = None
    recovery_count: int = 0
    recovery_failures: int = 0
    last_error: Optional[str] = None


@dataclass
class SDRplayRecovery:
    """Manages automatic recovery of SDRplay API service.

    When the SDRplay API service gets stuck (common after device errors,
    application crashes, or USB issues), this class can automatically
    restart the service to recover.
    """

    enabled: bool = True
    cooldown_seconds: float = 60.0  # Minimum time between restarts
    max_restarts_per_hour: int = 5

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _stats: RecoveryStats = field(default_factory=RecoveryStats)
    _restart_times: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._system = platform.system()
        logger.info(
            f"SDRplay recovery initialized: enabled={self.enabled}, "
            f"cooldown={self.cooldown_seconds}s, max_per_hour={self.max_restarts_per_hour}"
        )

    @property
    def stats(self) -> RecoveryStats:
        """Get recovery statistics."""
        return self._stats

    def _prune_restart_times(self) -> None:
        """Remove restart times older than 1 hour."""
        cutoff = time.time() - 3600
        self._restart_times = [t for t in self._restart_times if t > cutoff]

    def can_restart(self) -> tuple[bool, str]:
        """Check if a restart is allowed.

        Returns:
            Tuple of (allowed, reason)
        """
        if not self.enabled:
            return False, "Recovery disabled"

        with self._lock:
            now = time.time()

            # Check cooldown
            if self._stats.last_recovery_attempt:
                elapsed = now - self._stats.last_recovery_attempt
                if elapsed < self.cooldown_seconds:
                    remaining = self.cooldown_seconds - elapsed
                    return False, f"Cooldown active ({remaining:.0f}s remaining)"

            # Check rate limit
            self._prune_restart_times()
            if len(self._restart_times) >= self.max_restarts_per_hour:
                return False, f"Rate limit reached ({self.max_restarts_per_hour}/hour)"

            return True, "OK"

    def restart_service(self, reason: str = "unknown") -> bool:
        """Restart the SDRplay API service.

        Args:
            reason: Description of why restart is needed (for logging)

        Returns:
            True if restart succeeded, False otherwise
        """
        allowed, deny_reason = self.can_restart()
        if not allowed:
            logger.warning(f"SDRplay service restart denied: {deny_reason}")
            return False

        with self._lock:
            self._stats.last_recovery_attempt = time.time()
            self._stats.recovery_count += 1

        logger.warning(f"Restarting SDRplay API service (reason: {reason})")

        try:
            success = self._do_restart()
            if success:
                with self._lock:
                    self._stats.last_recovery_success = time.time()
                    self._restart_times.append(time.time())
                    self._stats.last_error = None
                logger.info("SDRplay API service restarted successfully")
                # Give the service time to initialize
                time.sleep(2)
                return True
            else:
                with self._lock:
                    self._stats.recovery_failures += 1
                    self._stats.last_error = "Restart command failed"
                return False
        except Exception as e:
            with self._lock:
                self._stats.recovery_failures += 1
                self._stats.last_error = str(e)
            logger.error(f"Failed to restart SDRplay service: {e}")
            return False

    def _do_restart(self) -> bool:
        """Execute platform-specific service restart."""
        if self._system == "Darwin":
            return self._restart_macos()
        elif self._system == "Linux":
            return self._restart_linux()
        elif self._system == "Windows":
            return self._restart_windows()
        else:
            logger.error(f"Unsupported platform for SDRplay recovery: {self._system}")
            return False

    def _restart_macos(self) -> bool:
        """Restart SDRplay service on macOS using launchctl."""
        # Try launchctl kickstart first (preferred, restarts cleanly)
        try:
            result = subprocess.run(
                ["sudo", "-n", "launchctl", "kickstart", "-kp", "system/com.sdrplay.service"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                logger.info(f"launchctl kickstart succeeded: {result.stdout.strip()}")
                return True
            else:
                logger.warning(f"launchctl kickstart failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.error("launchctl kickstart timed out")
        except FileNotFoundError:
            logger.error("sudo not found")

        # Fallback: try killall (service should auto-restart via launchd)
        try:
            result = subprocess.run(
                ["sudo", "-n", "killall", "sdrplay_apiService"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                logger.info("Killed sdrplay_apiService (should auto-restart)")
                time.sleep(3)  # Wait for launchd to restart it
                return True
        except Exception as e:
            logger.error(f"killall fallback failed: {e}")

        return False

    def _restart_linux(self) -> bool:
        """Restart SDRplay service on Linux using systemctl."""
        # Check if systemctl is available
        if not shutil.which("systemctl"):
            logger.error("systemctl not found")
            return False

        try:
            result = subprocess.run(
                ["sudo", "-n", "systemctl", "restart", "sdrplayService"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                logger.info("systemctl restart succeeded")
                return True
            else:
                logger.error(f"systemctl restart failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("systemctl restart timed out")
            return False

    def _restart_windows(self) -> bool:
        """Restart SDRplay service on Windows."""
        try:
            # Stop service
            subprocess.run(
                ["net", "stop", "SDRplayService"],
                capture_output=True,
                timeout=30,
            )
            time.sleep(1)
            # Start service
            result = subprocess.run(
                ["net", "start", "SDRplayService"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Windows service restart failed: {e}")
            return False


# Global singleton instance
_recovery_instance: Optional[SDRplayRecovery] = None
_instance_lock = threading.Lock()


def get_recovery() -> SDRplayRecovery:
    """Get or create the global SDRplay recovery instance."""
    global _recovery_instance
    with _instance_lock:
        if _recovery_instance is None:
            _recovery_instance = SDRplayRecovery()
        return _recovery_instance


def configure_recovery(
    enabled: bool = True,
    cooldown_seconds: float = 60.0,
    max_restarts_per_hour: int = 5,
) -> SDRplayRecovery:
    """Configure the global SDRplay recovery instance."""
    global _recovery_instance
    with _instance_lock:
        _recovery_instance = SDRplayRecovery(
            enabled=enabled,
            cooldown_seconds=cooldown_seconds,
            max_restarts_per_hour=max_restarts_per_hour,
        )
        return _recovery_instance


def attempt_recovery(reason: str = "unknown") -> bool:
    """Attempt to recover the SDRplay service.

    This is the main entry point for triggering recovery from other modules.

    Args:
        reason: Description of why recovery is needed

    Returns:
        True if recovery was attempted and succeeded
    """
    recovery = get_recovery()
    return recovery.restart_service(reason)
