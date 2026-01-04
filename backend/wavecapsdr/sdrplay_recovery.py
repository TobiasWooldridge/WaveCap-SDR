"""SDRplay service recovery module.

Automatically detects and recovers from stuck SDRplay API service.

This module integrates with the health monitoring features in SoapySDRPlay3:
- Uses the sdrplay-service-restart script for consistent service management
- Exposes DeviceHealthStatus for monitoring from the driver level
- Provides both automatic and manual recovery options
"""

import logging
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DeviceHealthStatus(Enum):
    """Device health status - mirrors SoapySDRPlay3's DeviceHealthStatus.

    These states are reported by the driver's watchdog system.
    """

    HEALTHY = "healthy"  # Stream active, callbacks arriving normally
    WARNING = "warning"  # Minor issues (slow callbacks, high timeouts)
    STALE = "stale"  # Callbacks stopped arriving
    RECOVERING = "recovering"  # Recovery in progress
    SERVICE_UNRESPONSIVE = "service_unresponsive"  # API calls timing out
    DEVICE_REMOVED = "device_removed"  # USB device disconnected
    FAILED = "failed"  # Unrecoverable failure


@dataclass
class HealthInfo:
    """Detailed health information from SoapySDRPlay3 driver.

    Maps to the HealthInfo struct in SoapySDRPlay.hpp.
    """

    status: DeviceHealthStatus = DeviceHealthStatus.HEALTHY
    callback_count: int = 0
    callback_rate: float = 0.0  # callbacks per second
    consecutive_timeouts: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    last_error: Optional[str] = None
    last_healthy_time: Optional[float] = None


@dataclass
class WatchdogConfig:
    """Watchdog configuration - mirrors SoapySDRPlay3's WatchdogConfig.

    Can be passed to SoapySDR device to configure the driver's watchdog.
    """

    enabled: bool = True
    callback_timeout_ms: int = 2000  # Max time between callbacks before stale
    health_check_interval_ms: int = 500  # How often to check health
    max_recovery_attempts: int = 3  # Per session
    recovery_backoff_ms: int = 1000  # Initial backoff between attempts
    auto_recover: bool = True  # Automatic vs manual recovery
    restart_service_on_failure: bool = True  # Try to restart SDRplay service
    usb_reset_on_failure: bool = False  # Try USB power cycle (requires uhubctl)


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
        """Restart SDRplay service on macOS.

        Uses the sdrplay-service-restart script from SoapySDRPlay3 if available,
        which handles SIGHUP soft restart vs full restart, plist detection, and
        stale lock file cleanup. Falls back to direct launchctl/killall.
        """
        # Strategy 1: Try sdrplay-service-restart script (from SoapySDRPlay3)
        # This script is designed to be added to sudoers for passwordless execution
        for script_cmd in [
            "sdrplay-service-restart",  # If installed in PATH
            str(Path(__file__).parent.parent.parent.parent / "SoapySDRPlay3" / "scripts" / "sdrplay-service-restart"),
        ]:
            try:
                # Try without sudo first (in case script has correct permissions)
                result = subprocess.run(
                    [script_cmd, "--force"],  # --force skips SIGHUP soft restart
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    logger.info(f"sdrplay-service-restart succeeded: {result.stdout.strip()}")
                    return True
                # Try with sudo
                result = subprocess.run(
                    ["sudo", "-n", script_cmd, "--force"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    logger.info(f"sdrplay-service-restart (sudo) succeeded: {result.stdout.strip()}")
                    return True
            except subprocess.TimeoutExpired:
                logger.warning(f"{script_cmd} timed out")
            except FileNotFoundError:
                continue  # Try next script location
            except Exception as e:
                logger.debug(f"{script_cmd} error: {e}")

        # Strategy 2: Try the full fix script (includes USB power cycling)
        fix_script = Path(__file__).parent.parent / "scripts" / "fix-sdrplay-full.sh"
        if fix_script.exists():
            try:
                logger.info(f"Running fix script: {fix_script}")
                result = subprocess.run(
                    ["sudo", "-n", str(fix_script)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    logger.info(f"Fix script succeeded: {result.stdout.strip()}")
                    return True
                else:
                    logger.warning(f"Fix script failed (rc={result.returncode}): {result.stderr}")
            except subprocess.TimeoutExpired:
                logger.error("Fix script timed out")
            except FileNotFoundError:
                logger.error("sudo not found")
            except Exception as e:
                logger.warning(f"Fix script error: {e}")

        # Strategy 3: launchctl kickstart (restarts cleanly)
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

        # Strategy 4: killall (service should auto-restart via launchd)
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
        """Restart SDRplay service on Linux.

        Uses the sdrplay-service-restart script from SoapySDRPlay3 if available,
        falls back to systemctl.
        """
        # Strategy 1: Try sdrplay-service-restart script (from SoapySDRPlay3)
        for script_cmd in [
            "sdrplay-service-restart",  # If installed in PATH
            str(Path(__file__).parent.parent.parent.parent / "SoapySDRPlay3" / "scripts" / "sdrplay-service-restart"),
        ]:
            try:
                result = subprocess.run(
                    ["sudo", "-n", script_cmd, "--force"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    logger.info(f"sdrplay-service-restart succeeded: {result.stdout.strip()}")
                    return True
            except subprocess.TimeoutExpired:
                logger.warning(f"{script_cmd} timed out")
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.debug(f"{script_cmd} error: {e}")

        # Strategy 2: systemctl
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


def check_service_responsive(timeout_seconds: float = 5.0) -> bool:
    """Check if the SDRplay service is responsive.

    Uses SoapySDRUtil --find with a timeout to detect if the service is responsive.
    This is a quick health check that doesn't require opening a device.

    Args:
        timeout_seconds: Maximum time to wait for service response

    Returns:
        True if service responded within timeout
    """
    try:
        result = subprocess.run(
            ["SoapySDRUtil", "--find=sdrplay"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        # Success if we got any output (even if no devices found)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning(f"SoapySDRUtil timed out after {timeout_seconds}s - service may be unresponsive")
        return False
    except FileNotFoundError:
        logger.warning("SoapySDRUtil not found in PATH")
        return True  # Can't check, assume OK
    except Exception as e:
        logger.error(f"Error checking service health: {e}")
        return False


def get_driver_health_info(device: Any) -> Optional[HealthInfo]:
    """Get health information from a SoapySDR device.

    The SoapySDRPlay3 driver provides health monitoring via getHealthInfo().

    Args:
        device: SoapySDR device object

    Returns:
        HealthInfo if the device supports health monitoring, None otherwise
    """
    try:
        # Check if device has the health monitoring API
        if not hasattr(device, "getHealthInfo"):
            return None

        raw_info = device.getHealthInfo()

        # Map driver's DeviceHealthStatus enum to our enum
        status_map = {
            0: DeviceHealthStatus.HEALTHY,  # Healthy
            1: DeviceHealthStatus.WARNING,  # Warning
            2: DeviceHealthStatus.STALE,  # Stale
            3: DeviceHealthStatus.RECOVERING,  # Recovering
            4: DeviceHealthStatus.SERVICE_UNRESPONSIVE,  # ServiceUnresponsive
            5: DeviceHealthStatus.DEVICE_REMOVED,  # DeviceRemoved
            6: DeviceHealthStatus.FAILED,  # Failed
        }

        status_code = getattr(raw_info, "status", 0)
        status = status_map.get(status_code, DeviceHealthStatus.HEALTHY)

        return HealthInfo(
            status=status,
            callback_count=getattr(raw_info, "callbackCount", 0),
            callback_rate=getattr(raw_info, "callbackRate", 0.0),
            consecutive_timeouts=getattr(raw_info, "consecutiveTimeouts", 0),
            recovery_attempts=getattr(raw_info, "recoveryAttempts", 0),
            successful_recoveries=getattr(raw_info, "successfulRecoveries", 0),
            last_error=getattr(raw_info, "lastError", None),
            last_healthy_time=None,  # Not directly exposed in C++ API
        )
    except Exception as e:
        logger.debug(f"Failed to get driver health info: {e}")
        return None


def configure_driver_watchdog(device: Any, config: WatchdogConfig) -> bool:
    """Configure the watchdog settings on a SoapySDR device.

    The SoapySDRPlay3 driver accepts watchdog configuration via device settings:
    - watchdog_enabled
    - callback_timeout_ms
    - max_recovery_attempts
    - auto_recover
    - restart_service_on_failure
    - usb_reset_on_failure

    Args:
        device: SoapySDR device object
        config: Watchdog configuration

    Returns:
        True if configuration was applied, False if not supported
    """
    try:
        # SoapySDRPlay3 exposes watchdog config via device settings
        settings_map = {
            "watchdog_enabled": str(config.enabled).lower(),
            "callback_timeout_ms": str(config.callback_timeout_ms),
            "max_recovery_attempts": str(config.max_recovery_attempts),
            "auto_recover": str(config.auto_recover).lower(),
            "restart_service_on_failure": str(config.restart_service_on_failure).lower(),
            "usb_reset_on_failure": str(config.usb_reset_on_failure).lower(),
        }

        applied = 0
        for key, value in settings_map.items():
            try:
                device.writeSetting(key, value)
                applied += 1
            except Exception:
                pass  # Setting may not be supported on this driver

        if applied > 0:
            logger.info(f"Driver watchdog configured ({applied} settings applied)")
            return True
        return False
    except Exception as e:
        logger.debug(f"Failed to configure driver watchdog: {e}")
        return False


def read_driver_watchdog_config(device: Any) -> Optional[WatchdogConfig]:
    """Read the current watchdog configuration from a SoapySDR device.

    Args:
        device: SoapySDR device object

    Returns:
        WatchdogConfig if settings are available, None otherwise
    """
    try:
        def read_bool(key: str, default: bool) -> bool:
            try:
                val = device.readSetting(key)
                return val.lower() == "true"
            except Exception:
                return default

        def read_int(key: str, default: int) -> int:
            try:
                return int(device.readSetting(key))
            except Exception:
                return default

        return WatchdogConfig(
            enabled=read_bool("watchdog_enabled", True),
            callback_timeout_ms=read_int("callback_timeout_ms", 2000),
            max_recovery_attempts=read_int("max_recovery_attempts", 3),
            auto_recover=read_bool("auto_recover", True),
            restart_service_on_failure=read_bool("restart_service_on_failure", True),
            usb_reset_on_failure=read_bool("usb_reset_on_failure", False),
        )
    except Exception as e:
        logger.debug(f"Failed to read driver watchdog config: {e}")
        return None


def trigger_driver_recovery(device: Any) -> bool:
    """Trigger recovery on a SoapySDR device.

    The SoapySDRPlay3 driver provides triggerRecovery() for manual recovery.

    Args:
        device: SoapySDR device object

    Returns:
        True if recovery was triggered, False if not supported or failed
    """
    try:
        if not hasattr(device, "triggerRecovery"):
            return False

        return bool(device.triggerRecovery())
    except Exception as e:
        logger.error(f"Failed to trigger driver recovery: {e}")
        return False
