"""Collection CRUD — plain sqlite3, no ORM."""
import json
import sqlite3
from datetime import date

from db.connection import get_conn


# ── helpers ──────────────────────────────────────────────────────────────────

def _row_to_card(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ('colours', 'colour_identity', 'keywords', 'creature_types'):
        d[key] = json.loads(d.get(key) or '[]')
    return d


def _row_to_copy(row: sqlite3.Row, card: dict, location: dict | None) -> dict:
    return {
        'id': row['id'],
        'card': card,
        'foil': bool(row['foil']),
        'etched': bool(row['etched']),
        'condition': row['condition'],
        'purchase_date': row['purchase_date'],
        'purchase_price': row['purchase_price'],
        'purchase_source': row['purchase_source'],
        'notes': row['notes'],
        'current_location': location,
    }


# ── copies ────────────────────────────────────────────────────────────────────

def list_copies(location_id: int | None = None) -> list[dict]:
    conn = get_conn()
    try:
        if location_id is not None:
            rows = conn.execute("""
                SELECT oc.*, cp.location_id
                FROM owned_copies oc
                LEFT JOIN card_placements cp ON cp.copy_id = oc.id
                WHERE cp.location_id = ?
            """, (location_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT oc.*, cp.location_id
                FROM owned_copies oc
                LEFT JOIN card_placements cp ON cp.copy_id = oc.id
            """).fetchall()

        result = []
        for row in rows:
            card_row = conn.execute('SELECT * FROM cards WHERE id = ?', (row['card_id'],)).fetchone()
            card = _row_to_card(card_row) if card_row else {}
            loc = None
            if row['location_id']:
                loc_row = conn.execute('SELECT * FROM locations WHERE id = ?', (row['location_id'],)).fetchone()
                loc = dict(loc_row) if loc_row else None
            result.append(_row_to_copy(row, card, loc))
        return result
    finally:
        conn.close()


def add_copy(card_id: str, location_id: int, foil: bool = False, etched: bool = False,
             condition: str = 'NM', purchase_date: str | None = None,
             purchase_price: float | None = None, purchase_source: str | None = None,
             notes: str | None = None) -> dict:
    conn = get_conn()
    try:
        cur = conn.execute("""
            INSERT INTO owned_copies
                (card_id, foil, etched, condition, purchase_date, purchase_price, purchase_source, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (card_id, int(foil), int(etched), condition, purchase_date, purchase_price, purchase_source, notes))
        copy_id = cur.lastrowid

        conn.execute('INSERT INTO card_placements (copy_id, location_id) VALUES (?, ?)', (copy_id, location_id))
        conn.execute("""
            INSERT INTO card_movements (copy_id, from_location_id, to_location_id, reason)
            VALUES (?, NULL, ?, 'Added to collection')
        """, (copy_id, location_id))
        conn.commit()

        return get_copy(copy_id)
    finally:
        conn.close()


def get_copy(copy_id: int) -> dict | None:
    conn = get_conn()
    try:
        row = conn.execute("""
            SELECT oc.*, cp.location_id
            FROM owned_copies oc
            LEFT JOIN card_placements cp ON cp.copy_id = oc.id
            WHERE oc.id = ?
        """, (copy_id,)).fetchone()
        if row is None:
            return None
        card_row = conn.execute('SELECT * FROM cards WHERE id = ?', (row['card_id'],)).fetchone()
        card = _row_to_card(card_row) if card_row else {}
        loc = None
        if row['location_id']:
            loc_row = conn.execute('SELECT * FROM locations WHERE id = ?', (row['location_id'],)).fetchone()
            loc = dict(loc_row) if loc_row else None
        return _row_to_copy(row, card, loc)
    finally:
        conn.close()


def update_copy(copy_id: int, condition: str | None = None,
                notes: str | None = None, purchase_price: float | None = None) -> dict | None:
    conn = get_conn()
    try:
        if condition is not None:
            conn.execute('UPDATE owned_copies SET condition = ? WHERE id = ?', (condition, copy_id))
        if notes is not None:
            conn.execute('UPDATE owned_copies SET notes = ? WHERE id = ?', (notes, copy_id))
        if purchase_price is not None:
            conn.execute('UPDATE owned_copies SET purchase_price = ? WHERE id = ?', (purchase_price, copy_id))
        conn.commit()
        return get_copy(copy_id)
    finally:
        conn.close()


def delete_copy(copy_id: int) -> bool:
    conn = get_conn()
    try:
        conn.execute('DELETE FROM card_movements WHERE copy_id = ?', (copy_id,))
        conn.execute('DELETE FROM card_placements WHERE copy_id = ?', (copy_id,))
        cur = conn.execute('DELETE FROM owned_copies WHERE id = ?', (copy_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ── locations ─────────────────────────────────────────────────────────────────

def list_locations() -> list[dict]:
    conn = get_conn()
    try:
        return [dict(r) for r in conn.execute('SELECT * FROM locations WHERE archived = 0').fetchall()]
    finally:
        conn.close()


def create_location(name: str, loc_type: str = 'storage') -> dict:
    conn = get_conn()
    try:
        cur = conn.execute('INSERT INTO locations (name, type) VALUES (?, ?)', (name, loc_type))
        conn.commit()
        row = conn.execute('SELECT * FROM locations WHERE id = ?', (cur.lastrowid,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def update_location(location_id: int, name: str | None = None, archived: bool | None = None) -> dict | None:
    conn = get_conn()
    try:
        if name is not None:
            conn.execute('UPDATE locations SET name = ? WHERE id = ?', (name, location_id))
        if archived is not None:
            conn.execute('UPDATE locations SET archived = ? WHERE id = ?', (int(archived), location_id))
        conn.commit()
        row = conn.execute('SELECT * FROM locations WHERE id = ?', (location_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── movements ─────────────────────────────────────────────────────────────────

def move_card(copy_id: int, to_location_id: int, reason: str = '') -> dict | None:
    conn = get_conn()
    try:
        placement = conn.execute(
            'SELECT location_id FROM card_placements WHERE copy_id = ?', (copy_id,)
        ).fetchone()
        from_location_id = placement['location_id'] if placement else None

        if placement:
            conn.execute(
                'UPDATE card_placements SET location_id = ?, placed_at = datetime("now") WHERE copy_id = ?',
                (to_location_id, copy_id),
            )
        else:
            conn.execute('INSERT INTO card_placements (copy_id, location_id) VALUES (?, ?)', (copy_id, to_location_id))

        cur = conn.execute("""
            INSERT INTO card_movements (copy_id, from_location_id, to_location_id, reason)
            VALUES (?, ?, ?, ?)
        """, (copy_id, from_location_id, to_location_id, reason))
        conn.commit()

        movement = conn.execute('SELECT * FROM card_movements WHERE id = ?', (cur.lastrowid,)).fetchone()
        from_loc = conn.execute('SELECT name FROM locations WHERE id = ?', (from_location_id,)).fetchone() if from_location_id else None
        to_loc = conn.execute('SELECT name FROM locations WHERE id = ?', (to_location_id,)).fetchone()
        return {
            'id': movement['id'],
            'copy_id': copy_id,
            'from_location': from_loc['name'] if from_loc else None,
            'to_location': to_loc['name'] if to_loc else None,
            'moved_at': movement['moved_at'],
            'reason': movement['reason'],
        }
    finally:
        conn.close()


def get_copy_history(copy_id: int) -> list[dict]:
    conn = get_conn()
    try:
        movements = conn.execute(
            'SELECT * FROM card_movements WHERE copy_id = ? ORDER BY moved_at', (copy_id,)
        ).fetchall()
        result = []
        for m in movements:
            from_loc = conn.execute('SELECT name FROM locations WHERE id = ?', (m['from_location_id'],)).fetchone() if m['from_location_id'] else None
            to_loc = conn.execute('SELECT name FROM locations WHERE id = ?', (m['to_location_id'],)).fetchone()
            result.append({
                'id': m['id'],
                'copy_id': m['copy_id'],
                'from_location': from_loc['name'] if from_loc else None,
                'to_location': to_loc['name'] if to_loc else None,
                'moved_at': m['moved_at'],
                'reason': m['reason'],
            })
        return result
    finally:
        conn.close()
