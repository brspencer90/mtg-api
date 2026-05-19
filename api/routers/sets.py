"""
Sets router — list available sets and stream full-set ingest via SSE.
"""
import json
import time
from datetime import datetime, timedelta

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from db.connection import get_conn
from mtg_func import get_card_info

router = APIRouter(prefix='/api', tags=['sets'])


@router.get('/sets')
def list_sets() -> list[dict]:
    """Return recent expansion sets from Scryfall."""
    resp = requests.get('https://api.scryfall.com/sets', timeout=10)
    data = resp.json().get('data', [])
    cutoff = datetime.now() - timedelta(days=365 * 3)
    return [
        {'id': s['code'], 'name': s['name'], 'released_at': s['released_at']}
        for s in data
        if s.get('set_type') in ('expansion', 'draft_innovation', 'masters')
        and datetime.strptime(s['released_at'], '%Y-%m-%d') > cutoff
    ]


@router.get('/collections/{collection_id}/ingest/set')
def ingest_full_set(collection_id: int, set_id: str):
    """
    SSE stream — fetches a full set from Scryfall card by card.
    Emits: data: {"done": N, "total": M, "name": "Card Name"}
    Final:  data: {"complete": true}

    Frontend opens an EventSource and updates a progress bar.
    """
    conn = get_conn()
    coll = conn.execute('SELECT id FROM collections WHERE id = ?', (collection_id,)).fetchone()
    conn.close()
    if coll is None:
        raise HTTPException(status_code=404, detail='Collection not found')

    def event_stream():
        set_json = requests.get(f'https://api.scryfall.com/sets/{set_id}', timeout=10).json()
        if set_json.get('object') == 'error':
            yield f'data: {json.dumps({"error": "Set not found"})}\n\n'
            return

        total = set_json['card_count']
        done = 0
        card_id = 1
        consecutive_errors = 0

        db = get_conn()
        try:
            while done < total and consecutive_errors <= 10:
                card = get_card_info(card_id, set_id)
                if isinstance(card, list):
                    _upsert_card_to_db(db, card, set_id)
                    _link_card_to_collection(db, collection_id, set_id, str(card_id))
                    db.commit()
                    done += 1
                    consecutive_errors = 0
                    name = card[0] if card else ''
                    yield f'data: {json.dumps({"done": done, "total": total, "name": name})}\n\n'
                else:
                    consecutive_errors += 1

                card_id += 1

            yield f'data: {json.dumps({"complete": True, "done": done, "total": total})}\n\n'
        finally:
            db.close()

    return StreamingResponse(event_stream(), media_type='text/event-stream')


def _upsert_card_to_db(conn, card_list: list, set_id: str) -> int:
    """Upsert a card row from a get_card_info() return list. Returns the row id."""
    from constants import Constants as c
    import json as _json

    # card_list matches Constants.col order
    col = c.col
    idx = {name: i for i, name in enumerate(col)}

    name = card_list[idx['Name']]
    collector_no = str(card_list[idx['Collector No.']])
    mana_cost = str(card_list[idx['Mana Cost']])
    cmc = card_list[idx['CMC']]
    power = str(card_list[idx['Power']])
    toughness = str(card_list[idx['Toughness']])
    type_line = str(card_list[idx['Type']])
    oracle_text = str(card_list[idx['Text']])
    colours = _json.dumps(card_list[idx['Colours']])
    keywords = _json.dumps(card_list[idx['Keywords']])
    rarity = str(card_list[idx['Rarity']])
    price = card_list[idx['Price']]
    price_std = card_list[idx['Price STD']]
    price_foil = card_list[idx['Price Foil']]
    price_etched = card_list[idx['Price Etched']]

    cur = conn.execute("""
        INSERT INTO cards
            (collector_no, set_id, name, mana_cost, cmc, power, toughness,
             type_line, oracle_text, rarity, colours, keywords,
             price, price_std, price_foil, price_etched)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(collector_no, set_id) DO UPDATE SET
            price     = excluded.price,
            price_std = excluded.price_std,
            price_foil = excluded.price_foil,
            price_etched = excluded.price_etched
    """, (collector_no, set_id, name, mana_cost, cmc, power, toughness,
          type_line, oracle_text, rarity, colours, keywords,
          price, price_std, price_foil, price_etched))
    # Return the existing or newly inserted id
    row = conn.execute(
        'SELECT id FROM cards WHERE collector_no = ? AND set_id = ?', (collector_no, set_id)
    ).fetchone()
    return row['id']


def _link_card_to_collection(conn, collection_id: int, set_id: str, collector_no: str):
    row = conn.execute(
        'SELECT id FROM cards WHERE collector_no = ? AND set_id = ?', (collector_no, set_id)
    ).fetchone()
    if row is None:
        return
    conn.execute("""
        INSERT OR IGNORE INTO collection_cards (collection_id, card_id)
        VALUES (?, ?)
    """, (collection_id, row['id']))
