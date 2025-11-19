# Activate venv, install deps (if not installed), and run the FastAPI app
param(
    [switch]$install
)

$venv = Join-Path $PSScriptRoot '..' '.venv' 'Scripts' 'Activate.ps1'
if ($install) {
    python -m venv ..\.venv
    ..\.venv\Scripts\Activate.ps1
    pip install -r ..\requirements.txt
}

..\.venv\Scripts\Activate.ps1
uvicorn src.backend.app:app --reload --port 8000
