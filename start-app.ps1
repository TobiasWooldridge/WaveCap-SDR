#!/usr/bin/env pwsh
# WaveCap-SDR startup script for Windows/PowerShell
# Starts the server with sensible defaults
#
# Optional environment variables:
#   $env:HOST="0.0.0.0"              # Bind address (default: 0.0.0.0)
#   $env:PORT=8087                   # Port number (default: 8087)
#   $env:DRIVER="soapy"              # SDR driver (default: soapy)
#   $env:DEVICE_ARGS="driver=..."    # Specific device arguments (optional)
#   $env:CONFIG="path/to/config.yaml" # Config file path (optional)
#
# Examples:
#   .\start-app.ps1                                    # Start with defaults
#   $env:HOST="127.0.0.1"; $env:PORT=8088; .\start-app.ps1  # Custom host/port
#   $env:DEVICE_ARGS="driver=rtlsdr"; .\start-app.ps1       # Specific device

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ScriptDir "backend"
$VenvDir = Join-Path $BackendDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

# Helper function for colored output
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

Write-ColorOutput "WaveCap-SDR Startup" "Green"
Write-ColorOutput "================================" "Green"

# Check if we're in the right directory
if (-not (Test-Path $BackendDir)) {
    Write-ColorOutput "Error: backend/ directory not found" "Red"
    Write-ColorOutput "Please run this script from the WaveCap-SDR root directory" "Red"
    exit 1
}

Set-Location $BackendDir

# Set up virtual environment if needed
if (-not (Test-Path $VenvPython)) {
    Write-ColorOutput "Setting up Python virtual environment..." "Yellow"
    python -m venv $VenvDir
    & $VenvPython -m pip install --upgrade pip --quiet
    Write-ColorOutput "Installing dependencies..." "Green"
    & $VenvPython -m pip install fastapi uvicorn httpx websockets pyyaml numpy scipy --quiet
}

# Default values
$Host_Addr = if ($env:HOST) { $env:HOST } else { "0.0.0.0" }
$Port_Num = if ($env:PORT) { $env:PORT } else { "8087" }
$Driver = if ($env:DRIVER) { $env:DRIVER } else { "soapy" }

# Set environment variables for WaveCap-SDR config
# These override config file settings
$env:WAVECAPSDR__SERVER__BIND_ADDRESS = $Host_Addr
$env:WAVECAPSDR__SERVER__PORT = $Port_Num
$env:WAVECAPSDR__DEVICE__DRIVER = $Driver
if ($env:DEVICE_ARGS) {
    $env:WAVECAPSDR__DEVICE__DEVICE_ARGS = $env:DEVICE_ARGS
}

Write-Host ""
Write-ColorOutput "Configuration:" "Cyan"
Write-Host "  Host: $Host_Addr"
Write-Host "  Port: $Port_Num"
Write-Host "  Driver: $Driver"
if ($env:DEVICE_ARGS) { Write-Host "  Device: $env:DEVICE_ARGS" }
if ($env:CONFIG) { Write-Host "  Config: $env:CONFIG" }
Write-Host ""
Write-ColorOutput "Starting WaveCap-SDR server..." "Green"
Write-Host "Web UI will be available at: http://${Host_Addr}:${Port_Num}/"
Write-Host "Press Ctrl+C to stop"
Write-Host ""

# Build command arguments
$CmdArgs = @(
    "-m", "wavecapsdr"
)

# Add optional config file if provided
if ($env:CONFIG) {
    $CmdArgs += @("--config", $env:CONFIG)
}

# Set PYTHONPATH and start the server (config comes from environment variables)
$env:PYTHONPATH = "."
& $VenvPython $CmdArgs
