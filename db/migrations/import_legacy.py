"""
One-time import: scans card data/*.txt files and populates the SQLite DB.

Filename convention: YYMMDD_SETID_description_words.txt
  250208_dft_sealed_cards.txt   → set_id=dft, date=2025-02-08, location="DFT Sealed Cards"
  260402_ecl_costco_booster.txt → set_id=ecl, date=2026-04-02, location="ECL Costco Booster"

Run from the project root:
  python -m db.migrations.import_legacy
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from db.connection import get_conn, init_db


def parse_filename(path: Path) -> tuple[str, str | None, str]:
    """Return (set_id, purchase_date_iso, location_name) from a data file path."""
    stem = path.stem  # e.g. '260402_ecl_costco_booster'
    parts = stem.split('_')

    date_str = parts[0]       # YYMMDD
    set_id = parts[1].lower() if len(parts) > 1 else 'unknown'
    desc_parts = parts[2:] if len(parts) > 2 else []

    try:
        purchase_date = date(
            year=2000 + int(date_str[:2]),
            month=int(date_str[2:4]),
            day=int(date_str[4:6]),
        ).isoformat()
    except (ValueError, IndexError):
        purchase_date = None

    desc = ' '.join(desc_parts).replace('_', ' ').title()
    location_name = f"{set_id.upper()} {desc}".strip()
    return set_id, purchase_date, location_name


def fetch_card(set_id: str, collector_no: str) -> dict | None:
    time.sleep(0.1)
    try:
        data = requests.get(
            f'https://api.scryfall.com/cards/{set_id}/{collector_no}',
            timeout=10,
        ).json()
        return data if data.get('object') != 'error' else None
    except Exception as exc:
        print(f'  [warn] {set_id}/{collector_no}: {exc}')
        return None


def card_type_from_json(card_json: dict) -> str:
    tl = card_json.get('type_line', '')
    if 'Creature' in tl:
        return 'Creature'
    if 'Land' in tl:
        return 'Land'
    return 'Non-Creature'


def upsert_card(conn, card_json: dict) -> str:
    """Insert or update a card row. Returns the Scryfall UUID."""
    scryfall_id = card_json['id']
    type_line = card_json.get('type_line', '')
    creature_types: list[str] = []
    if 'Creature' in type_line and '—' in type_line:
        creature_types = type_line.split('—')[-1].strip().split()

    conn.execute("""
        INSERT INTO cards
            (id, name, set_id, collector_no, cmc, mana_cost, card_type,
             colours, colour_identity, keywords, creature_types,
             rarity, text, power, toughness,
             price_std, price_foil, price_etched)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            price_std    = excluded.price_std,
            price_foil   = excluded.price_foil,
            price_etched = excluded.price_etched,
            last_updated = datetime('now')
    """, (
        scryfall_id,
        card_json.get('name', ''),
        card_json.get('set', ''),
        card_json.get('collector_number', ''),
        card_json.get('cmc'),
        card_json.get('mana_cost', ''),
        card_type_from_json(card_json),
        json.dumps(card_json.get('colors', [])),
        json.dumps(card_json.get('color_identity', [])),
        json.dumps(card_json.get('keywords', [])),
        json.dumps(creature_types),
        card_json.get('rarity', ''),
        card_json.get('oracle_text', ''),
        card_json.get('power', ''),
        card_json.get('toughness', ''),
        _safe_float(card_json.get('prices', {}).get('usd')),
        _safe_float(card_json.get('prices', {}).get('usd_foil')),
        _safe_float(card_json.get('prices', {}).get('usd_etched')),
    ))
    return scryfall_id


def _safe_float(val) -> float | None:
    try:
        return float(val) if val else None
    except (TypeError, ValueError):
        return None


def get_or_create_location(conn, name: str, loc_type: str) -> int:
    row = conn.execute('SELECT id FROM locations WHERE name = ?', (name,)).fetchone()
    if row:
        return row['id']
    cur = conn.execute(
        'INSERT INTO locations (name, type) VALUES (?, ?)', (name, loc_type)
    )
    return cur.lastrowid


def import_file(conn, txt_path: Path) -> int:
    set_id, purchase_date, location_name = parse_filename(txt_path)
    print(f'\n>> {txt_path.name}  [set={set_id}, date={purchase_date}, loc="{location_name}"]')

    loc_type = 'pool' if 'sealed' in txt_path.stem else 'storage'
    location_id = get_or_create_location(conn, location_name, loc_type)

    lines = [l.strip() for l in txt_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    imported = 0

    for raw_id in lines:
        foil = int(raw_id.endswith('f'))
        etched = int(raw_id.endswith('e'))
        collector_no = raw_id.rstrip('fe')

        card_json = fetch_card(set_id, collector_no)
        if card_json is None:
            print(f'  [skip] {set_id}/{collector_no}')
            continue

        card_id = upsert_card(conn, card_json)

        cur = conn.execute("""
            INSERT INTO owned_copies
                (card_id, foil, etched, condition, purchase_date, purchase_source)
            VALUES (?, ?, ?, 'NM', ?, ?)
        """, (card_id, foil, etched, purchase_date, location_name))
        copy_id = cur.lastrowid

        conn.execute(
            'INSERT INTO card_placements (copy_id, location_id) VALUES (?, ?)',
            (copy_id, location_id),
        )
        conn.execute("""
            INSERT INTO card_movements (copy_id, from_location_id, to_location_id, reason)
            VALUES (?, NULL, ?, 'Legacy import')
        """, (copy_id, location_id))

        imported += 1
        name = card_json.get('name', '?')
        suffix = ('f' if foil else '') + ('e' if etched else '')
        print(f'  [{imported}] {name} ({set_id} #{collector_no}{suffix})')

    conn.commit()
    return imported


def main():
    init_db()
    data_dir = ROOT / 'card data'
    txt_files = sorted(f for f in data_dir.glob('*.txt') if 'deck' not in f.stem.lower())

    if not txt_files:
        print('No .txt files found in card data/')
        return

    conn = get_conn()
    total = 0
    try:
        for path in txt_files:
            total += import_file(conn, path)
    finally:
        conn.close()

    print(f'\nDone — {total} cards imported across {len(txt_files)} files.')


if __name__ == '__main__':
    main()
