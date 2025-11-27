#!/usr/bin/env bash
# WaveCap-SDR Service Management Script
# Usage: ./manage-service.sh [install|uninstall|start|stop|restart|status|logs]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.wavecapsdr.server.plist"
PLIST_SRC="$SCRIPT_DIR/$PLIST_NAME"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME"
LOG_DIR="$PROJECT_DIR/logs"
SERVICE_SCRIPT="$SCRIPT_DIR/wavecapsdr-service.sh"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

check_prerequisites() {
    # Check if virtual environment exists
    if [ ! -x "$PROJECT_DIR/backend/.venv/bin/python" ]; then
        print_error "Virtual environment not found."
        echo "Please run ./start-app.sh once to set up the environment first."
        exit 1
    fi

    # Ensure service script is executable
    chmod +x "$SERVICE_SCRIPT"
}

install_service() {
    echo "Installing WaveCap-SDR service..."

    check_prerequisites

    # Create LaunchAgents directory if it doesn't exist
    mkdir -p "$HOME/Library/LaunchAgents"

    # Create logs directory
    mkdir -p "$LOG_DIR"

    # Ensure service script is executable
    chmod +x "$SERVICE_SCRIPT"

    # Copy plist to LaunchAgents
    cp "$PLIST_SRC" "$PLIST_DST"

    # Load the service
    launchctl load "$PLIST_DST"

    print_status "Service installed and started"
    echo ""
    echo "The service will:"
    echo "  - Start automatically on login"
    echo "  - Restart automatically if it crashes"
    echo "  - Listen on http://0.0.0.0:8087/"
    echo ""
    echo "Commands:"
    echo "  ./manage-service.sh status   - Check service status"
    echo "  ./manage-service.sh logs     - View logs"
    echo "  ./manage-service.sh stop     - Stop the service"
    echo "  ./manage-service.sh start    - Start the service"
}

uninstall_service() {
    echo "Uninstalling WaveCap-SDR service..."

    if [ -f "$PLIST_DST" ]; then
        # Unload the service (ignore errors if not loaded)
        launchctl unload "$PLIST_DST" 2>/dev/null || true

        # Remove the plist
        rm -f "$PLIST_DST"

        print_status "Service uninstalled"
    else
        print_warn "Service is not installed"
    fi
}

start_service() {
    echo "Starting WaveCap-SDR service..."

    if [ ! -f "$PLIST_DST" ]; then
        print_error "Service is not installed. Run: ./manage-service.sh install"
        exit 1
    fi

    launchctl start com.wavecapsdr.server
    sleep 2
    show_status
}

stop_service() {
    echo "Stopping WaveCap-SDR service..."

    if [ ! -f "$PLIST_DST" ]; then
        print_error "Service is not installed"
        exit 1
    fi

    launchctl stop com.wavecapsdr.server
    print_status "Service stopped"
}

restart_service() {
    echo "Restarting WaveCap-SDR service..."
    stop_service
    sleep 2
    start_service
}

show_status() {
    echo "WaveCap-SDR Service Status"
    echo "=========================="

    if [ ! -f "$PLIST_DST" ]; then
        print_warn "Service is not installed"
        return
    fi

    # Check if service is running
    if launchctl list | grep -q "com.wavecapsdr.server"; then
        # Get PID
        PID=$(launchctl list | grep "com.wavecapsdr.server" | awk '{print $1}')
        if [ "$PID" != "-" ] && [ -n "$PID" ]; then
            print_status "Service is running (PID: $PID)"

            # Check if web server is responding
            if command -v curl &>/dev/null; then
                if curl -s -o /dev/null -w "%{http_code}" "http://localhost:8087/api/v1/health" 2>/dev/null | grep -q "200"; then
                    print_status "Web server is responding on port 8087"
                else
                    print_warn "Web server not responding yet (may still be starting)"
                fi
            fi
        else
            print_warn "Service is loaded but not running"
        fi
    else
        print_warn "Service is not running"
    fi

    echo ""
    echo "Log files:"
    echo "  stdout: $LOG_DIR/wavecapsdr-stdout.log"
    echo "  stderr: $LOG_DIR/wavecapsdr-stderr.log"
    echo "  app:    $LOG_DIR/wavecapsdr.log"
}

show_logs() {
    echo "WaveCap-SDR Logs (last 50 lines)"
    echo "================================"

    if [ -f "$LOG_DIR/wavecapsdr.log" ]; then
        tail -50 "$LOG_DIR/wavecapsdr.log"
    elif [ -f "$LOG_DIR/wavecapsdr-stdout.log" ]; then
        tail -50 "$LOG_DIR/wavecapsdr-stdout.log"
    else
        print_warn "No log files found yet"
    fi
}

follow_logs() {
    echo "Following WaveCap-SDR logs (Ctrl+C to stop)..."
    echo "=============================================="

    if [ -f "$LOG_DIR/wavecapsdr.log" ]; then
        tail -f "$LOG_DIR/wavecapsdr.log"
    elif [ -f "$LOG_DIR/wavecapsdr-stdout.log" ]; then
        tail -f "$LOG_DIR/wavecapsdr-stdout.log"
    else
        print_warn "No log files found yet"
        echo "Waiting for logs..."
        while [ ! -f "$LOG_DIR/wavecapsdr.log" ] && [ ! -f "$LOG_DIR/wavecapsdr-stdout.log" ]; do
            sleep 1
        done
        if [ -f "$LOG_DIR/wavecapsdr.log" ]; then
            tail -f "$LOG_DIR/wavecapsdr.log"
        else
            tail -f "$LOG_DIR/wavecapsdr-stdout.log"
        fi
    fi
}

show_help() {
    echo "WaveCap-SDR Service Manager"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  install     Install and start the service (runs on login)"
    echo "  uninstall   Stop and remove the service"
    echo "  start       Start the service"
    echo "  stop        Stop the service"
    echo "  restart     Restart the service"
    echo "  status      Show service status"
    echo "  logs        Show recent log output"
    echo "  follow      Follow log output in real-time"
    echo ""
    echo "Example:"
    echo "  $0 install    # Install and start service"
    echo "  $0 status     # Check if it's running"
    echo "  $0 logs       # View recent logs"
}

# Main
case "${1:-help}" in
    install)
        install_service
        ;;
    uninstall)
        uninstall_service
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    follow)
        follow_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
