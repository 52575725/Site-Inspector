# Install Ollama and pull required models
# Run as Administrator if Ollama is not yet installed

Write-Host "=== Ollama Setup ===" -ForegroundColor Cyan

# Check if Ollama is already installed
$ollamaInstalled = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaInstalled) {
    Write-Host "Ollama not found. Download from: https://ollama.com/download/windows" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After installing Ollama, re-run this script to pull models." -ForegroundColor White
    exit 1
}

Write-Host "[OK] Ollama is installed" -ForegroundColor Green

# Check if Ollama server is running
try {
    $null = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5
    Write-Host "[OK] Ollama server is running" -ForegroundColor Green
} catch {
    Write-Host "Starting Ollama server..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

# Pull models
Write-Host ""
Write-Host "Pulling models..." -ForegroundColor Yellow

$models = @("llama3.2", "nomic-embed-text")
foreach ($model in $models) {
    Write-Host "  Pulling $model..."
    ollama pull $model
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] $model" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $model" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "=== Ollama Setup Complete ===" -ForegroundColor Cyan
Write-Host "Models available:"
ollama list
