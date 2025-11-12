#!/usr/bin/env pwsh
# WaveCap-SDR startup script for PowerShell
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
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDir = Join-Path $ScriptDir "backend"
$VenvDir = Join-Path $BackendDir ".venv"

# Determine Python and Pip executables based on platform
$VenvPython = if ($IsWindows -or $env:OS -eq "Windows_NT") {
    Join-Path $VenvDir "Scripts\python.exe"
} else {
    Join-Path $VenvDir "bin/python"
}

$VenvPip = if ($IsWindows -or $env:OS -eq "Windows_NT") {
    Join-Path $VenvDir "Scripts\pip.exe"
} else {
    Join-Path $VenvDir "bin/pip"
}

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

    # Find python3 or python command
    $PythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
    if (-not $PythonCmd) {
        $PythonCmd = Get-Command python -ErrorAction SilentlyContinue
    }
    if (-not $PythonCmd) {
        Write-ColorOutput "Error: Python 3 not found. Please install Python 3." "Red"
        exit 1
    }

    & $PythonCmd.Source -m venv --system-site-packages $VenvDir
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

# Attempt to refresh SDRplay service so devices enumerate cleanly.
# Note: This is typically only relevant on Linux/Unix systems
$RestartScript = Join-Path $ScriptDir "restart-sdrplay.sh"
if (Test-Path $RestartScript) {
    if (-not ($IsWindows -or $env:OS -eq "Windows_NT")) {
        try {
            & bash $RestartScript --non-interactive 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Refreshed sdrplay service."
            } else {
                Write-Host "Warning: Could not auto-restart sdrplay (sudo password likely required). Run ./restart-sdrplay.sh manually if needed."
            }
        } catch {
            Write-Host "Warning: Could not auto-restart sdrplay. Run ./restart-sdrplay.sh manually if needed."
        }
    }
}

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
