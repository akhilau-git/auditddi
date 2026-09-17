"""Portable reviewed-compound registry for self-hosted research deployments."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class CompoundRegistry:
    def __init__(self, state_dir: Path) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = state_dir / 'auditddi-registry.sqlite3'
        with self._connect() as connection:
            connection.execute(
                '''
                CREATE TABLE IF NOT EXISTS compounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    external_id TEXT,
                    smiles TEXT NOT NULL,
                    evidence_notes TEXT,
                    created_at INTEGER NOT NULL
                )
                '''
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def add(self, name: str, external_id: str | None, smiles: str, evidence_notes: str | None) -> dict:
        with self._connect() as connection:
            cursor = connection.execute(
                'INSERT INTO compounds(name, external_id, smiles, evidence_notes, created_at) VALUES (?, ?, ?, ?, ?)',
                (name.strip(), (external_id or '').strip() or None, smiles.strip(), (evidence_notes or '').strip() or None, int(time.time())),
            )
            compound_id = cursor.lastrowid
        return self.get(compound_id)

    def get(self, compound_id: int) -> dict:
        with self._connect() as connection:
            row = connection.execute('SELECT * FROM compounds WHERE id = ?', (compound_id,)).fetchone()
        if row is None:
            raise KeyError(compound_id)
        return dict(row)

    def list(self, limit: int = 100) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT id, name, external_id, smiles, evidence_notes, created_at FROM compounds ORDER BY id DESC LIMIT ?',
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
