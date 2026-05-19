"""Collection CRUD — plain sqlite3, no ORM."""
import json
import sqlite3
import time

import requests

from db.connection import get_conn
from db.migrations.import_legacy import upsert_card


# ── helpers ──────────────────────────────────────────────────────────────────

def _row_to_flat(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ('colours', 'colour_identity', 'keywords', 'creature_types'):
        if key in d:
            d[key] = json.loads(d.get(key) or '[]')
    return d


# ── copies (paginated + filtered) ─────────────────────────────────────────────

_SORT_COLS = {
    'name':       'c.name',
    'set_id':     'c.set_id',
    'card_type':  'c.card_type',
    'rarity':     "CASE c.rarity WHEN 'mythic' THEN 4 WHEN 'rare' THEN 3 WHEN 'uncommon' THEN 2 ELSE 1 END",
    'cmc':        'c.cmc',
    'price':      'COALESCE(CASE WHEN oc.foil=1 THEN c.price_foil ELSE NULL END, c.price_std)',
    'condition':  'oc.condition',
    'location':   'l.name',
    'collector_no': 'CAST(c.collector_no AS INTEGER)',
}


def list_copies(
    search: str | None = None,
    set_id: str | None = None,
    location_id: int | None = None,
    rarity: str | None = None,
    card_type: str | None = None,
    color: str | None = None,
    sort_by: str = 'name',
    sort_dir: str = 'asc',
    page: int = 1,
    page_size: int = 50,
) -> dict:
    """Return paginated, filtered owned copies as flat dicts."""
    conditions: list[str] = []
    params: list = []

    if search:
        conditions.append("c.name LIKE ?")
        params.append(f'%{search}%')
    if set_id:
        conditions.append("c.set_id = ?")
        params.append(set_id.lower())
    if location_id is not None:
        conditions.append("cp.location_id = ?")
        params.append(location_id)
    if rarity:
        conditions.append("c.rarity = ?")
        params.append(rarity)
    if card_type:
        conditions.append("c.card_type = ?")
        params.append(card_type)
    if color:
        # JSON array contains the color letter
        conditions.append("c.colours LIKE ?")
        params.append(f'%"{color}"%')

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    conn = get_conn()
    try:
        total = conn.execute(
            f"""SELECT COUNT(*) FROM owned_copies oc
                JOIN cards c ON c.id = oc.card_id
                LEFT JOIN card_placements cp ON cp.copy_id = oc.id
                {where}""",
            params,
        ).fetchone()[0]

        offset = (page - 1) * page_size
        rows = conn.execute(
            f"""SELECT
                  oc.id, oc.foil, oc.etched, oc.condition,
                  oc.purchase_date, oc.purchase_price, oc.purchase_source, oc.notes,
                  c.id   AS card_id,   c.name,       c.set_id,
                  c.collector_no,      c.mana_cost,  c.cmc,
                  c.card_type,         c.colours,    c.rarity,
                  c.price_std,         c.price_foil, c.price_etched,
                  cp.location_id,
                  l.name AS location_name,  l.type AS location_type
                FROM owned_copies oc
                JOIN cards c ON c.id = oc.card_id
                LEFT JOIN card_placements cp ON cp.copy_id = oc.id
                LEFT JOIN locations l ON l.id = cp.location_id
                {where}
                ORDER BY {_SORT_COLS.get(sort_by, 'c.name')} {'DESC' if sort_dir == 'desc' else 'ASC'}, c.name ASC
                LIMIT ? OFFSET ?""",
            params + [page_size, offset],
        ).fetchall()

        return {
            'total': total,
            'page': page,
            'page_size': page_size,
            'copies': [_row_to_flat(r) for r in rows],
        }
    finally:
        conn.close()


def ensure_card(scryfall_id: str | None = None,
                set_id: str | None = None,
                collector_no: str | None = None) -> dict | None:
    """
    Make sure a card row exists in the DB. Fetches from Scryfall if needed.
    Returns the card dict or None if not found.
    """
    conn = get_conn()
    try:
        if scryfall_id:
            row = conn.execute('SELECT * FROM cards WHERE id = ?', (scryfall_id,)).fetchone()
            if row:
                return _row_to_flat(row)
            # Fetch by UUID
            time.sleep(0.05)
            resp = requests.get(f'https://api.scryfall.com/cards/{scryfall_id}', timeout=10).json()
        elif set_id and collector_no:
            row = conn.execute(
                'SELECT * FROM cards WHERE set_id = ? AND collector_no = ?',
                (set_id.lower(), collector_no),
            ).fetchone()
            if row:
                return _row_to_flat(row)
            time.sleep(0.05)
            resp = requests.get(
                f'https://api.scryfall.com/cards/{set_id.lower()}/{collector_no}', timeout=10
            ).json()
        else:
            return None

        if resp.get('object') == 'error':
            return None

        card_id = upsert_card(conn, resp)
        conn.commit()
        row = conn.execute('SELECT * FROM cards WHERE id = ?', (card_id,)).fetchone()
        return _row_to_flat(row) if row else None
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
            SELECT oc.id, oc.foil, oc.etched, oc.condition,
                   oc.purchase_date, oc.purchase_price, oc.purchase_source, oc.notes,
                   c.id AS card_id, c.name, c.set_id, c.collector_no, c.card_type,
                   c.colours, c.rarity, c.price_std, c.price_foil,
                   cp.location_id, l.name AS location_name, l.type AS location_type
            FROM owned_copies oc
            JOIN cards c ON c.id = oc.card_id
            LEFT JOIN card_placements cp ON cp.copy_id = oc.id
            LEFT JOIN locations l ON l.id = cp.location_id
            WHERE oc.id = ?
        """, (copy_id,)).fetchone()
        return _row_to_flat(row) if row else None
    finally:
        conn.close()


def _refresh_card_price(conn, card_id: str) -> None:
    """Fetch current prices from Scryfall and update the cards table."""
    try:
        time.sleep(0.05)
        resp = requests.get(f'https://api.scryfall.com/cards/{card_id}', timeout=10).json()
        if resp.get('object') == 'error':
            return
        prices = resp.get('prices', {})
        def _f(v):
            try: return float(v) if v else None
            except (TypeError, ValueError): return None
        conn.execute("""
            UPDATE cards
               SET price_std    = ?,
                   price_foil   = ?,
                   price_etched = ?,
                   last_updated = datetime('now')
             WHERE id = ?
        """, (_f(prices.get('usd')), _f(prices.get('usd_foil')), _f(prices.get('usd_etched')), card_id))
    except Exception:
        pass  # price refresh is best-effort; don't fail the edit


def update_copy(copy_id: int, condition: str | None = None, notes: str | None = None,
                purchase_price: float | None = None, foil: bool | None = None,
                etched: bool | None = None, purchase_date: str | None = None,
                purchase_source: str | None = None) -> dict | None:
    conn = get_conn()
    try:
        fields = {
            'condition': condition, 'notes': notes,
            'purchase_price': purchase_price, 'purchase_date': purchase_date,
            'purchase_source': purchase_source,
        }
        for col, val in fields.items():
            if val is not None:
                conn.execute(f'UPDATE owned_copies SET {col} = ? WHERE id = ?', (val, copy_id))
        if foil is not None:
            conn.execute('UPDATE owned_copies SET foil = ? WHERE id = ?', (int(foil), copy_id))
        if etched is not None:
            conn.execute('UPDATE owned_copies SET etched = ? WHERE id = ?', (int(etched), copy_id))

        # Refresh price from Scryfall
        row = conn.execute('SELECT card_id FROM owned_copies WHERE id = ?', (copy_id,)).fetchone()
        if row:
            _refresh_card_price(conn, row['card_id'])

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

_TYPE_ORDER = "CASE type WHEN 'pool' THEN 0 WHEN 'deck' THEN 1 WHEN 'storage' THEN 2 ELSE 3 END"

def list_locations(include_archived: bool = False) -> list[dict]:
    conn = get_conn()
    try:
        where = '' if include_archived else 'WHERE archived = 0'
        rows = conn.execute(
            f'SELECT * FROM locations {where} ORDER BY {_TYPE_ORDER}, name'
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def ensure_default_locations() -> None:
    """Create basic physical locations if they don't already exist."""
    defaults = [('Card Pool', 'pool'), ('Storage', 'storage'), ('Trade Pile', 'trade')]
    conn = get_conn()
    try:
        for name, loc_type in defaults:
            if not conn.execute('SELECT 1 FROM locations WHERE name = ?', (name,)).fetchone():
                conn.execute('INSERT INTO locations (name, type) VALUES (?, ?)', (name, loc_type))
        conn.commit()
    finally:
        conn.close()


def create_location(name: str, loc_type: str = 'storage') -> dict:
    conn = get_conn()
    try:
        cur = conn.execute('INSERT INTO locations (name, type) VALUES (?, ?)', (name, loc_type))
        conn.commit()
        return dict(conn.execute('SELECT * FROM locations WHERE id = ?', (cur.lastrowid,)).fetchone())
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

        conn.execute("""
            INSERT INTO card_movements (copy_id, from_location_id, to_location_id, reason)
            VALUES (?, ?, ?, ?)
        """, (copy_id, from_location_id, to_location_id, reason))
        conn.commit()
        return get_copy(copy_id)
    finally:
        conn.close()


def get_copy_history(copy_id: int) -> list[dict]:
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT cm.*, fl.name as from_name, tl.name as to_name
            FROM card_movements cm
            LEFT JOIN locations fl ON fl.id = cm.from_location_id
            JOIN  locations tl ON tl.id = cm.to_location_id
            WHERE cm.copy_id = ?
            ORDER BY cm.moved_at
        """, (copy_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ── filter option helpers ──────────────────────────────────────────────────────

def list_owned_sets() -> list[str]:
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT DISTINCT c.set_id
            FROM owned_copies oc JOIN cards c ON c.id = oc.card_id
            ORDER BY c.set_id
        """).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()
