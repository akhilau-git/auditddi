"""Small first-party authentication store for self-hosted AuditDDI deployments.

This is intentionally limited to a portable single-instance deployment. Larger
installations should use their organisation's OIDC/SSO gateway instead.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any


PBKDF2_ITERATIONS = 600_000
SESSION_TTL_SECONDS = 8 * 60 * 60


class AuthStore:
    def __init__(self, state_dir: Path) -> None:
        state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = state_dir / 'auditddi-auth.sqlite3'
        with self._connect() as connection:
            connection.executescript(
                '''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                '''
            )
            columns = {
                row['name'] for row in connection.execute('PRAGMA table_info(users)')
            }
            if 'is_active' not in columns:
                connection.execute(
                    'ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1'
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _hash_password(password: str, salt: bytes | None = None) -> str:
        salt = salt or secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, PBKDF2_ITERATIONS)
        return f'{salt.hex()}${digest.hex()}'

    @staticmethod
    def _password_matches(password: str, encoded: str) -> bool:
        try:
            salt_hex, digest_hex = encoded.split('$', 1)
            expected = AuthStore._hash_password(password, bytes.fromhex(salt_hex))
            return hmac.compare_digest(expected, encoded)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode('utf-8')).hexdigest()

    def has_users(self) -> bool:
        with self._connect() as connection:
            return connection.execute('SELECT 1 FROM users LIMIT 1').fetchone() is not None

    def create_user(self, email: str, password: str, role: str) -> dict[str, Any]:
        normalized_email = email.strip().lower()
        if len(normalized_email) < 3 or '@' not in normalized_email:
            raise ValueError('A valid email address is required.')
        if len(password) < 12:
            raise ValueError('Password must contain at least 12 characters.')
        with self._connect() as connection:
            cursor = connection.execute(
                'INSERT INTO users(email, password_hash, role, created_at) VALUES (?, ?, ?, ?)',
                (normalized_email, self._hash_password(password), role, int(time.time())),
            )
            return {
                'id': cursor.lastrowid, 'email': normalized_email,
                'role': role, 'is_active': True,
            }

    def list_users(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                'SELECT id, email, role, is_active, created_at FROM users ORDER BY email'
            ).fetchall()
        return [dict(row) for row in rows]

    def set_user_active(self, user_id: int, is_active: bool) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                'UPDATE users SET is_active = ? WHERE id = ?', (int(is_active), user_id)
            )
            if cursor.rowcount != 1:
                raise ValueError('User account was not found.')
            if not is_active:
                connection.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))

    def authenticate(self, email: str, password: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            user = connection.execute(
                'SELECT id, email, password_hash, role, is_active FROM users WHERE email = ?',
                (email.strip().lower(),),
            ).fetchone()
        if (user is None or not bool(user['is_active'])
                or not self._password_matches(password, user['password_hash'])):
            return None
        return {
            'id': user['id'], 'email': user['email'], 'role': user['role'],
            'is_active': True,
        }

    def issue_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        with self._connect() as connection:
            connection.execute('DELETE FROM sessions WHERE expires_at < ?', (int(time.time()),))
            connection.execute(
                'INSERT INTO sessions(token_hash, user_id, expires_at) VALUES (?, ?, ?)',
                (self._token_hash(token), user_id, int(time.time()) + SESSION_TTL_SECONDS),
            )
        return token

    def session_user(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        with self._connect() as connection:
            row = connection.execute(
                '''
                SELECT users.id, users.email, users.role, users.is_active
                FROM sessions JOIN users ON users.id = sessions.user_id
                WHERE sessions.token_hash = ? AND sessions.expires_at > ? AND users.is_active = 1
                ''',
                (self._token_hash(token), int(time.time())),
            ).fetchone()
        return dict(row) if row is not None else None

    def revoke_session(self, token: str | None) -> None:
        if token:
            with self._connect() as connection:
                connection.execute('DELETE FROM sessions WHERE token_hash = ?', (self._token_hash(token),))
