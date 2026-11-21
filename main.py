"""
Main entry point alias for FedSafe-X FastAPI Backend.
Allows running via either:
    uvicorn main:app --reload --port 8000
or
    uvicorn app:app --reload --port 8000
"""

from app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
