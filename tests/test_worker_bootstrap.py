"""Tests for viz_authoring.worker_bootstrap.init_worker_subprocess."""
from __future__ import annotations

import logging
import sqlite3

import pytest

from viz_authoring.worker_bootstrap import init_worker_subprocess


@pytest.fixture
def isolated_logging(monkeypatch):
    """Stash + restore root logger config so test mutations don't
    leak across tests / sessions."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    root.handlers[:] = original_handlers
    root.setLevel(original_level)


class TestInitWorkerSubprocess:
    """init_worker_subprocess is the standard bootstrap for worker
    subprocesses. Tests run in-process (no actual subprocess spawn)
    since the function's job is just to do the setup steps."""

    def test_returns_logger_and_connection(self, tmp_path, isolated_logging):
        db = tmp_path / 'test.db'
        log, conn = init_worker_subprocess('test.worker', str(db))
        try:
            assert isinstance(log, logging.Logger)
            assert log.name == 'test.worker'
            assert isinstance(conn, sqlite3.Connection)
        finally:
            conn.close()

    def test_sets_busy_timeout(self, tmp_path, isolated_logging):
        db = tmp_path / 'test.db'
        log, conn = init_worker_subprocess(
            'test.worker', str(db), busy_timeout_ms=7777)
        try:
            # SQLite returns 0 if not set, else the timeout in ms
            result = conn.execute('PRAGMA busy_timeout').fetchone()
            assert result[0] == 7777
        finally:
            conn.close()

    def test_default_busy_timeout(self, tmp_path, isolated_logging):
        db = tmp_path / 'test.db'
        log, conn = init_worker_subprocess('test.worker', str(db))
        try:
            result = conn.execute('PRAGMA busy_timeout').fetchone()
            assert result[0] == 30_000
        finally:
            conn.close()

    def test_does_not_run_schema_migrations(self, tmp_path, isolated_logging):
        """viz_authoring is schema-agnostic — caller is responsible for
        running app-specific schema_init on the returned connection."""
        db = tmp_path / 'test.db'
        log, conn = init_worker_subprocess('test.worker', str(db))
        try:
            # Connection is empty — no tables created
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            assert tables == []
        finally:
            conn.close()

    def test_silence_pil_sets_pil_level(self, tmp_path, isolated_logging):
        db = tmp_path / 'test.db'
        log, conn = init_worker_subprocess(
            'test.worker', str(db), silence_pil=True)
        try:
            assert logging.getLogger('PIL').level == logging.WARNING
        finally:
            conn.close()

    def test_silence_pil_off_by_default(self, tmp_path, isolated_logging,
                                          monkeypatch):
        # Reset PIL logger level so we measure the function's effect
        logging.getLogger('PIL').setLevel(logging.NOTSET)
        db = tmp_path / 'test.db'
        log, conn = init_worker_subprocess('test.worker', str(db))
        try:
            # Default — function did not touch PIL level
            assert logging.getLogger('PIL').level == logging.NOTSET
        finally:
            conn.close()

    def test_creates_db_file(self, tmp_path, isolated_logging):
        db = tmp_path / 'nonexistent.db'
        assert not db.exists()
        log, conn = init_worker_subprocess('test.worker', str(db))
        try:
            assert db.exists()
        finally:
            conn.close()

    def test_handler_reset_clears_root(self, tmp_path, isolated_logging):
        # Add a custom handler to root, verify it gets cleared
        root = logging.getLogger()
        marker_handler = logging.NullHandler()
        root.addHandler(marker_handler)
        db = tmp_path / 'test.db'
        log, conn = init_worker_subprocess('test.worker', str(db))
        try:
            assert marker_handler not in root.handlers
        finally:
            conn.close()

    def test_handler_reset_namespaces_param(self, tmp_path, isolated_logging):
        # Add handlers to multiple logger namespaces, verify caller-
        # specified ones get cleared
        logger_a = logging.getLogger('myapp_a')
        logger_b = logging.getLogger('myapp_b')
        ha = logging.NullHandler()
        hb = logging.NullHandler()
        logger_a.addHandler(ha)
        logger_b.addHandler(hb)
        db = tmp_path / 'test.db'
        log, conn = init_worker_subprocess(
            'test.worker', str(db),
            handler_reset_namespaces=('myapp_a', None))
        try:
            assert ha not in logger_a.handlers
            # myapp_b not in namespaces list — should remain
            assert hb in logger_b.handlers
        finally:
            conn.close()
            # Cleanup
            logger_b.removeHandler(hb)
