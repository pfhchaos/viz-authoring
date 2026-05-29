"""Standard subprocess bootstrap for background workers.

Each worker subprocess (spawned via `multiprocessing.Process(target=...)`)
needs the same setup: clear inherited logging handlers, reconfigure
stdout logging, nice the process to background priority, optionally
silence noisy third-party loggers (PIL is common for workers that
touch images), open an app-specific SQLite connection with a long
busy_timeout. This module centralizes that pattern.

`init_worker_subprocess` returns `(log, conn)` and is intentionally
schema-agnostic — running app-specific schema migrations on the
connection is the caller's responsibility, since viz_authoring
doesn't know about your tables.

Pattern (from inside a worker subprocess `_main` function):

    from viz_authoring import init_worker_subprocess

    def _my_worker_main(db_path, stop_event):
        log, conn = init_worker_subprocess(
            'my_app.worker_name', db_path)
        my_schema_init(conn)  # ← caller does app-specific setup
        try:
            while not stop_event.is_set():
                # do work
                pass
        finally:
            conn.close()
"""
from __future__ import annotations

import logging
import os
import sqlite3


_DEFAULT_LOG_FORMAT = '%(asctime)s %(name)s %(levelname)s %(message)s'
_DEFAULT_DATEFMT = '%H:%M:%S'
_DEFAULT_BUSY_TIMEOUT_MS = 30_000


def init_worker_subprocess(logger_name: str,
                            db_path: str,
                            busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
                            silence_pil: bool = False,
                            handler_reset_namespaces: tuple[str | None, ...] = (None,),
                            ) -> tuple[logging.Logger, sqlite3.Connection]:
    """Bootstrap a background-worker subprocess.

    Steps:
      1. Clear inherited logging handlers (subprocess fork inherits the
         parent's handler list — we reset so this process's basicConfig
         actually takes effect). Clears the root logger by default;
         additional named-namespace roots can be passed in
         `handler_reset_namespaces` (e.g. `('my_app', None)` to clear
         both `my_app.*` and root).
      2. Configure stdout logging at INFO with timestamp + level + name
      3. Optionally silence PIL's INFO logs
      4. Nice the process to background priority (level 19)
      5. Open SQLite connection with `busy_timeout_ms` PRAGMA

    NOTE: does NOT run app-specific schema migrations. The caller is
    responsible for whatever `_ensure_schema(conn)` equivalent their
    app has — viz_authoring doesn't know about your tables.

    Returns:
        (log, conn) — the worker-named logger and an open SQLite
        connection. Caller owns `conn` (responsible for `conn.close()`
        in its own try/finally, typically as part of the worker's
        graceful shutdown).
    """
    # Subprocess fork inherits parent's handlers; clear so the
    # subprocess's basicConfig actually takes effect.
    for name in handler_reset_namespaces:
        logging.getLogger(name).handlers.clear()
    logging.basicConfig(level=logging.INFO,
                        format=_DEFAULT_LOG_FORMAT,
                        datefmt=_DEFAULT_DATEFMT)
    if silence_pil:
        logging.getLogger('PIL').setLevel(logging.WARNING)
    log = logging.getLogger(logger_name)

    # Lowest user-priority. Workers run only when interactive work
    # isn't holding the CPU.
    try:
        os.nice(19)
    except OSError:
        pass  # already at lowest priority or not permitted

    conn = sqlite3.connect(db_path)
    conn.execute(f'PRAGMA busy_timeout={busy_timeout_ms}')

    return log, conn
