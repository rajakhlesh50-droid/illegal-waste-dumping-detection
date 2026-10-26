# Launch Waste Detection System in project virtual environment
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir

if (Test-Path ".\venv\Scripts\python.exe") {
    Write-Host "Starting WasteAI using Virtual Environment..." -ForegroundColor Green
    & ".\venv\Scripts\python.exe" app.py
} else {
    Write-Host "Running with default python..." -ForegroundColor Yellow
    python app.py
}

