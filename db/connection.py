"""sqlite3 connection factory for the MTG collection database."""
import sqlite3
from pathlib import Path

from db.schema import SCHEMA_SQL

DB_PATH = Path(__file__).parent / 'mtg.db'


def get_conn(path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a sqlite3 connection with row_factory set to Row."""
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def init_db(path: Path = DB_PATH) -> None:
    """Create all tables if they don't exist yet. Safe to call repeatedly."""
    conn = get_conn(path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
