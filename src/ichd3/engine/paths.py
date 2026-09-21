"""Repo-Wurzel-Auflösung — eine Quelle für alle asset-/runs-relativen Pfade.

Reihenfolge: ICHD3_ROOT (explizit, z. B. /app im Container) > Suche nach
pyproject.toml aufwärts vom Paket (editable install im Repo) > cwd (Fallback,
z. B. wenn das Paket regulär installiert ist und man im Repo arbeitet).
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("ICHD3_ROOT")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()
