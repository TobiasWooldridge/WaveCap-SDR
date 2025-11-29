#!/bin/bash
# SDRplay Service Fix Script
# Restarts the SDRplay API service when device enumeration hangs
#
# Usage: ./fix_sdrplay.sh [--check|--fix|--setup-sudoers]
#
# To enable passwordless sudo for this script, run:
#   sudo ./fix_sdrplay.sh --setup-sudoers

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

detect_os() {
    case "$(uname -s)" in
        Darwin*)  echo "macos" ;;
        Linux*)   echo "linux" ;;
        MINGW*|MSYS*|CYGWIN*) echo "windows" ;;
        *)        echo "unknown" ;;
    esac
}

check_sdrplay_usb() {
    log_info "Checking for SDRplay USB device..."
    local os=$(detect_os)

    case "$os" in
        macos)
            if system_profiler SPUSBDataType 2>/dev/null | grep -q "1df7"; then
                log_info "SDRplay USB device found"
                return 0
            fi
            ;;
        linux)
            if lsusb 2>/dev/null | grep -qi "1df7\|sdrplay"; then
                log_info "SDRplay USB device found"
                return 0
            fi
            ;;
    esac

    log_error "SDRplay USB device not found"
    return 1
}

check_sdrplay_service() {
    log_info "Checking SDRplay service status..."
    local os=$(detect_os)

    case "$os" in
        macos)
            if pgrep -f sdrplay_apiService > /dev/null 2>&1; then
                local pid=$(pgrep -f sdrplay_apiService)
                log_info "SDRplay service running (PID: $pid)"
                return 0
            fi
            ;;
        linux)
            if systemctl is-active --quiet sdrplayService 2>/dev/null; then
                log_info "SDRplay service running (systemd)"
                return 0
            elif pgrep -f sdrplay_apiService > /dev/null 2>&1; then
                local pid=$(pgrep -f sdrplay_apiService)
                log_info "SDRplay service running (PID: $pid)"
                return 0
            fi
            ;;
    esac

    log_warn "SDRplay service not running"
    return 1
}

test_enumeration() {
    log_info "Testing SoapySDR device enumeration (5s timeout)..."

    SoapySDRUtil --find 2>&1 &
    local pid=$!

    sleep 5

    if ps -p $pid > /dev/null 2>&1; then
        kill $pid 2>/dev/null || true
        log_error "Enumeration STUCK - service needs restart"
        return 1
    else
        if wait $pid 2>/dev/null; then
            log_info "Enumeration completed successfully"
            return 0
        else
            log_warn "Enumeration completed with errors"
            return 0
        fi
    fi
}

restart_service_macos() {
    log_info "Restarting SDRplay service on macOS..."

    # Try launchctl kickstart first (preferred - kills and restarts, returns new PID)
    local new_pid
    new_pid=$(sudo -n /bin/launchctl kickstart -kp system/com.sdrplay.service 2>/dev/null)
    if [ -n "$new_pid" ]; then
        log_info "Service restarted via launchctl kickstart (new PID: $new_pid)"
        return 0
    fi

    # Fall back to launchctl stop/start
    if sudo -n launchctl stop com.sdrplay.apiservice 2>/dev/null; then
        sleep 2
        sudo -n launchctl start com.sdrplay.apiservice 2>/dev/null
        log_info "Service restarted via launchctl stop/start"
        return 0
    fi

    # Fall back to killall (requires sudo password if not in sudoers)
    log_info "Using killall (service will auto-restart)"
    sudo killall sdrplay_apiService 2>/dev/null || true
    sleep 2
}

restart_service_linux() {
    log_info "Restarting SDRplay service on Linux..."

    # Try systemctl first
    if command -v systemctl &> /dev/null && systemctl list-units --type=service | grep -q sdrplay; then
        sudo systemctl restart sdrplayService
        log_info "Service restarted via systemctl"
    else
        # Fall back to killall
        log_info "Using killall"
        sudo killall sdrplay_apiService 2>/dev/null || true
        sleep 1
        # Try to find and start the service binary
        for path in /usr/local/bin/sdrplay_apiService /opt/sdrplay/bin/sdrplay_apiService /usr/bin/sdrplay_apiService; do
            if [ -x "$path" ]; then
                sudo "$path" &
                log_info "Service started from $path"
                break
            fi
        done
    fi
}

restart_service() {
    local os=$(detect_os)

    case "$os" in
        macos)
            restart_service_macos
            ;;
        linux)
            restart_service_linux
            ;;
        windows)
            log_error "Windows: Run 'Restart-Service SDRplayService' in PowerShell as Administrator"
            return 1
            ;;
        *)
            log_error "Unknown operating system"
            return 1
            ;;
    esac

    sleep 2
    check_sdrplay_service
}

setup_sudoers_macos() {
    cat << 'EOF'
# Add the following to /etc/sudoers.d/sdrplay to enable passwordless SDRplay service restart
# Run: sudo visudo -f /etc/sudoers.d/sdrplay

# Allow admin group to restart SDRplay service without password
# kickstart is preferred - it kills and restarts in one command, returning the new PID
%admin ALL=(ALL) NOPASSWD: /bin/launchctl kickstart -kp system/com.sdrplay.service
%admin ALL=(ALL) NOPASSWD: /bin/launchctl stop com.sdrplay.apiservice
%admin ALL=(ALL) NOPASSWD: /bin/launchctl start com.sdrplay.apiservice
%admin ALL=(ALL) NOPASSWD: /usr/bin/killall sdrplay_apiService
EOF
}

setup_sudoers_linux() {
    cat << 'EOF'
# Add the following to /etc/sudoers.d/sdrplay to enable passwordless SDRplay service restart
# Run: sudo visudo -f /etc/sudoers.d/sdrplay

# Allow plugdev group to restart SDRplay service without password
%plugdev ALL=(ALL) NOPASSWD: /bin/systemctl restart sdrplayService
%plugdev ALL=(ALL) NOPASSWD: /usr/bin/killall sdrplay_apiService
EOF
}

setup_sudoers() {
    local os=$(detect_os)

    echo ""
    echo "===== Sudoers Configuration for Passwordless SDRplay Service Restart ====="
    echo ""

    case "$os" in
        macos)
            setup_sudoers_macos
            ;;
        linux)
            setup_sudoers_linux
            ;;
        windows)
            echo "Windows: Add user to Administrators group or use Task Scheduler"
            ;;
        *)
            echo "Unknown OS"
            ;;
    esac

    echo ""
    echo "==========================================================================="
}

do_check() {
    echo ""
    echo "===== SDRplay Service Diagnostic ====="
    echo "OS: $(detect_os)"
    echo ""

    check_sdrplay_usb || true
    echo ""
    check_sdrplay_service || true
    echo ""
    test_enumeration || true
    echo ""
}

do_fix() {
    echo ""
    echo "===== SDRplay Service Fix ====="
    echo "OS: $(detect_os)"
    echo ""

    if ! check_sdrplay_usb; then
        log_error "Cannot fix - SDRplay device not connected"
        exit 1
    fi

    echo ""
    log_info "Restarting SDRplay service..."
    restart_service

    echo ""
    log_info "Verifying fix..."
    sleep 2

    if test_enumeration; then
        echo ""
        log_info "SUCCESS: SDRplay service fixed!"
    else
        echo ""
        log_error "Service still stuck. Try unplugging and replugging the USB device."
        exit 1
    fi
}

show_usage() {
    echo "SDRplay Service Fix Script"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --check          Check SDRplay service status and test enumeration"
    echo "  --fix            Restart SDRplay service to fix stuck enumeration"
    echo "  --setup-sudoers  Show sudoers config for passwordless restart"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --check       # Diagnose issues"
    echo "  $0 --fix         # Fix stuck service (requires sudo)"
    echo "  $0 --setup-sudoers | sudo tee /etc/sudoers.d/sdrplay"
}

# Main
case "${1:-}" in
    --check)
        do_check
        ;;
    --fix)
        do_fix
        ;;
    --setup-sudoers)
        setup_sudoers
        ;;
    -h|--help)
        show_usage
        ;;
    "")
        # Default: check then fix if needed
        do_check
        if ! test_enumeration 2>/dev/null; then
            echo ""
            read -p "Service appears stuck. Restart it? [y/N] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                do_fix
            fi
        fi
        ;;
    *)
        log_error "Unknown option: $1"
        show_usage
        exit 1
        ;;
esac
