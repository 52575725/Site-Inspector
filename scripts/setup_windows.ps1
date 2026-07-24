# Site Inspector - Windows Setup Script
# Run this once to set up all dependencies

Write-Host "=== Site Inspector Setup ===" -ForegroundColor Cyan
Write-Host ""

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "[OK] $pythonVersion"

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "[OK] Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "[OK] Virtual environment already exists" -ForegroundColor Green
}

# Activate and install dependencies
Write-Host ""
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]" --quiet
Write-Host "[OK] Python dependencies installed" -ForegroundColor Green

# Install Playwright browsers
Write-Host ""
Write-Host "Installing Playwright browsers..." -ForegroundColor Yellow
playwright install chromium
Write-Host "[OK] Playwright Chromium installed" -ForegroundColor Green

# Check Node.js (for Lighthouse and axe-core)
Write-Host ""
$nodeVersion = node --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] $nodeVersion" -ForegroundColor Green
    Write-Host "Installing Lighthouse and axe-core CLI..." -ForegroundColor Yellow
    npm install -g lighthouse @axe-core/cli 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Lighthouse and axe-core installed" -ForegroundColor Green
    } else {
        Write-Host "[WARN] Could not install Lighthouse/axe-core (npm may not be available)" -ForegroundColor Yellow
        Write-Host "       Performance and accessibility checks will use limited mode"
    }
} else {
    Write-Host "[WARN] Node.js not found. Install from https://nodejs.org/" -ForegroundColor Yellow
    Write-Host "       Lighthouse and axe-core checks will use limited fallback mode"
}

# Check Ollama
Write-Host ""
Write-Host "Checking Ollama..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -ErrorAction Stop
    Write-Host "[OK] Ollama server is running" -ForegroundColor Green
    Write-Host "Pulling recommended models..."
    ollama pull llama3.2 2>&1 | Out-Null
    ollama pull nomic-embed-text 2>&1 | Out-Null
    Write-Host "[OK] Models pulled (llama3.2, nomic-embed-text)" -ForegroundColor Green
} catch {
    Write-Host "[WARN] Ollama not running. Install from https://ollama.com/" -ForegroundColor Yellow
    Write-Host "       AI-powered features (alt text, content analysis) will use fallback mode"
}

# Initialize database
Write-Host ""
Write-Host "Initializing database..." -ForegroundColor Yellow
.\.venv\Scripts\python.exe -c "import asyncio; from src.storage.database import init_db; asyncio.run(init_db())"
Write-Host "[OK] Database initialized" -ForegroundColor Green

# Create data directories
Write-Host ""
Write-Host "Creating data directories..." -ForegroundColor Yellow
$dirs = @("data\scans", "data\screenshots", "data\reports", "data\site_sources")
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
Write-Host "[OK] Data directories created" -ForegroundColor Green

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Activate venv:  .\.venv\Scripts\Activate.ps1"
Write-Host "  2. Run a test scan: site-inspector scan run"
Write-Host "  3. View status:      site-inspector status"
Write-Host ""
Write-Host "Optional - Configure Google APIs:"
Write-Host "  site-inspector config --set google_credentials_path=C:\path\to\google-credentials.json"
Write-Host "  site-inspector config --set gsc_property=sc-domain:helinsilver.com"
