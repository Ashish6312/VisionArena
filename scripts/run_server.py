#!/usr/bin/env python
"""Run the FastAPI server.

Usage:
    python scripts/run_server.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from app.config import settings

if __name__ == "__main__":
    uvicorn.run("app.backend.main:app", host=settings.api_host, port=settings.api_port, reload=True)
