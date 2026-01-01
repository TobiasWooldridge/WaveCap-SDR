#!/bin/bash
#
# fix-sdr-devices.sh - Comprehensive SDR device reset script
#
# This script reliably resets all SDR devices by:
# 1. Killing all processes using SDR devices
# 2. Stopping the SDRplay API service
# 3. Resetting USB devices (if usbreset is available)
# 4. Restarting the SDRplay API service
# 5. Verifying all devices are available
#
# Usage: ./fix-sdr-devices.sh [--no-verify] [--quick]
#
# Requires sudo for SDRplay API restart
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MAX_RETRIES=3
SLEEP_SHORT=2
SLEEP_MEDIUM=5
SLEEP_LONG=10

# Parse arguments
VERIFY=true
QUICK=false
for arg in "$@"; do
    case $arg in
        --no-verify) VERIFY=false ;;
        --quick) QUICK=true; SLEEP_SHORT=1; SLEEP_MEDIUM=2; SLEEP_LONG=3 ;;
    esac
done

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Kill all SDR-related processes
kill_sdr_processes() {
    log_info "Killing SDR-related processes..."

    local killed=0

    # WaveCap-SDR
    if pgrep -f "wavecapsdr" > /dev/null 2>&1; then
        log_info "  Stopping WaveCap-SDR..."
        pkill -9 -f "wavecapsdr" 2>/dev/null || true
        killed=$((killed + 1))
    fi

    # SDRTrunk
    if pgrep -f "SDRTrunk" > /dev/null 2>&1; then
        log_info "  Stopping SDRTrunk..."
        pkill -9 -f "SDRTrunk" 2>/dev/null || true
        # Also try osascript for graceful quit
        osascript -e 'tell application "SDRTrunk" to quit' 2>/dev/null || true
        killed=$((killed + 1))
    fi

    # SoapySDR processes
    if pgrep -f "SoapySDR" > /dev/null 2>&1; then
        log_info "  Stopping SoapySDR processes..."
        pkill -9 -f "SoapySDR" 2>/dev/null || true
        killed=$((killed + 1))
    fi

    # rtl_* processes (rtl_test, rtl_fm, etc.)
    if pgrep -f "rtl_" > /dev/null 2>&1; then
        log_info "  Stopping rtl_* processes..."
        pkill -9 -f "rtl_" 2>/dev/null || true
        killed=$((killed + 1))
    fi

    # GQRX
    if pgrep -f "gqrx" > /dev/null 2>&1; then
        log_info "  Stopping GQRX..."
        pkill -9 -f "gqrx" 2>/dev/null || true
        killed=$((killed + 1))
    fi

    # SDR# (if running under Wine)
    if pgrep -f "SDRSharp" > /dev/null 2>&1; then
        log_info "  Stopping SDR#..."
        pkill -9 -f "SDRSharp" 2>/dev/null || true
        killed=$((killed + 1))
    fi

    # CubicSDR
    if pgrep -f "CubicSDR" > /dev/null 2>&1; then
        log_info "  Stopping CubicSDR..."
        pkill -9 -f "CubicSDR" 2>/dev/null || true
        killed=$((killed + 1))
    fi

    # Wait for processes to fully terminate
    sleep $SLEEP_SHORT

    # Verify no SDR processes remain
    local remaining=0
    for proc in wavecapsdr SDRTrunk SoapySDR rtl_ gqrx; do
        if pgrep -f "$proc" > /dev/null 2>&1; then
            remaining=$((remaining + 1))
            log_warn "  Process '$proc' still running, force killing..."
            pkill -9 -f "$proc" 2>/dev/null || true
        fi
    done

    if [ $remaining -gt 0 ]; then
        sleep $SLEEP_SHORT
    fi

    if [ $killed -gt 0 ]; then
        log_success "Killed $killed SDR process(es)"
    else
        log_success "No SDR processes were running"
    fi
}

# Step 2: Stop SDRplay API service
stop_sdrplay_api() {
    log_info "Stopping SDRplay API service..."

    if [ -f /Library/LaunchDaemons/com.sdrplay.service.plist ]; then
        # Try to stop the service
        if sudo launchctl stop com.sdrplay.service 2>/dev/null; then
            log_success "SDRplay API service stopped"
        else
            log_warn "SDRplay API service may not have been running"
        fi

        # Also unload it to ensure clean state
        sudo launchctl unload /Library/LaunchDaemons/com.sdrplay.service.plist 2>/dev/null || true

        sleep $SLEEP_SHORT
    else
        log_warn "SDRplay API service plist not found - skipping"
    fi
}

# Step 3: Reset USB devices using uhubctl power cycling
reset_usb_devices() {
    log_info "Resetting USB devices..."

    # Check if uhubctl is available (preferred method)
    if command -v uhubctl &> /dev/null; then
        # Power cycle all ports on smart USB hubs
        # uhubctl will find all controllable hubs automatically
        log_info "  Power cycling USB ports with uhubctl..."

        # Power off all controllable ports
        if sudo uhubctl -a off 2>/dev/null; then
            log_info "  USB ports powered off"
            sleep 2

            # Power on all controllable ports
            if sudo uhubctl -a on 2>/dev/null; then
                log_info "  USB ports powered on"
                sleep 3
                log_success "USB power cycle complete"
            else
                log_warn "  Failed to power on USB ports"
            fi
        else
            log_warn "  No controllable USB hubs found or uhubctl failed"
            log_info "  Trying to list available hubs..."
            sudo uhubctl 2>/dev/null | head -20 || true
        fi
    else
        log_warn "  uhubctl not available"
        log_info "  Install with: brew install uhubctl"
        log_info "  Skipping USB power cycle - devices may need manual reconnection"
    fi
}

# Step 4: Start SDRplay API service
start_sdrplay_api() {
    log_info "Starting SDRplay API service..."

    if [ -f /Library/LaunchDaemons/com.sdrplay.service.plist ]; then
        # Load and start the service
        sudo launchctl load /Library/LaunchDaemons/com.sdrplay.service.plist 2>/dev/null || true
        sleep $SLEEP_SHORT

        if sudo launchctl start com.sdrplay.service 2>/dev/null; then
            log_success "SDRplay API service started"
        else
            # Service might already be running after load
            if launchctl list 2>/dev/null | grep -q "sdrplay"; then
                log_success "SDRplay API service is running"
            else
                log_error "Failed to start SDRplay API service"
                return 1
            fi
        fi

        # Give the API time to fully initialize
        sleep $SLEEP_MEDIUM
    else
        log_warn "SDRplay API service plist not found - skipping"
    fi
}

# Step 5: Verify devices are available
verify_devices() {
    log_info "Verifying SDR devices..."

    local retry=0
    local devices_found=0
    local expected_sdrplay=2  # We expect 2 SDRplay devices
    local expected_rtlsdr=1   # We expect 1 RTL-SDR device

    while [ $retry -lt $MAX_RETRIES ]; do
        # Run SoapySDRUtil to find devices
        local output=$(SoapySDRUtil --find 2>/dev/null)

        # Count devices (grep -c returns 1 on no match, so we capture and default)
        local sdrplay_count
        local rtlsdr_count
        sdrplay_count=$(echo "$output" | grep -c "driver = sdrplay") || sdrplay_count=0
        rtlsdr_count=$(echo "$output" | grep -c "driver = rtlsdr") || rtlsdr_count=0

        devices_found=$((sdrplay_count + rtlsdr_count))

        log_info "  Found: $sdrplay_count SDRplay, $rtlsdr_count RTL-SDR device(s)"

        if [ "$sdrplay_count" -ge "$expected_sdrplay" ] && [ "$rtlsdr_count" -ge "$expected_rtlsdr" ]; then
            log_success "All expected devices found!"

            # Print device details
            echo ""
            echo "Available devices:"
            echo "$output" | grep -A3 "Found device" | head -30
            echo ""
            return 0
        fi

        retry=$((retry + 1))
        if [ $retry -lt $MAX_RETRIES ]; then
            log_warn "  Not all devices found, retrying in ${SLEEP_MEDIUM}s... (attempt $retry/$MAX_RETRIES)"
            sleep $SLEEP_MEDIUM
        fi
    done

    log_error "Failed to find all expected devices after $MAX_RETRIES attempts"
    log_info "  Found $devices_found device(s), expected $((expected_sdrplay + expected_rtlsdr))"

    # Show what we found anyway
    echo ""
    echo "Available devices:"
    SoapySDRUtil --find 2>/dev/null || echo "  (SoapySDRUtil failed)"
    echo ""

    return 1
}

# Main execution
main() {
    echo ""
    echo "========================================"
    echo "  SDR Device Reset Script"
    echo "========================================"
    echo ""

    # Check if running as root for certain operations
    if [ "$EUID" -ne 0 ]; then
        log_info "Note: Some operations require sudo (you may be prompted)"
    fi

    # Execute steps
    kill_sdr_processes
    echo ""

    stop_sdrplay_api
    echo ""

    reset_usb_devices
    echo ""

    # Wait for USB to settle
    log_info "Waiting for USB to settle..."
    sleep $SLEEP_LONG

    start_sdrplay_api
    echo ""

    if [ "$VERIFY" = true ]; then
        verify_devices
        local result=$?
        echo ""

        if [ $result -eq 0 ]; then
            log_success "SDR devices are ready!"
            echo ""
            echo "You can now start WaveCap-SDR or SDRTrunk."
        else
            log_error "Some devices may not be available"
            echo ""
            echo "Troubleshooting tips:"
            echo "  1. Check USB cable connections"
            echo "  2. Try unplugging and replugging the devices"
            echo "  3. Run this script again"
            echo "  4. Check system logs: log show --predicate 'eventMessage CONTAINS \"USB\"' --last 1m"
        fi

        return $result
    else
        log_success "Reset complete (verification skipped)"
    fi
}

# Run main function
main "$@"
