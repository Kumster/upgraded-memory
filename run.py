"""Simple runner to start Uvicorn from the project root.

Usage:
  . .\.venv\Scripts\Activate.ps1
  python run.py

This avoids import problems when running `py app.py` from inside the `backend/` folder.
"""
import uvicorn

if __name__ == '__main__':
    uvicorn.run('backend.app:app', host='127.0.0.1', port=8000, reload=True)
