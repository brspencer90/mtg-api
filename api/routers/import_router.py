"""
Deck import router — Moxfield URL preview and SSE-streaming save to DB.
"""
import json
import re
import time

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from db.connection import get_conn
from db.migrations.import_legacy import fetch_card, upsert_card

router = APIRouter(prefix='/api/import', tags=['import'])

_MOXFIELD_DECK_URL = 'https://api.moxfield.com/v2/decks/all/{}'


def _extract_deck_id(url: str) -> str:
    m = re.search(r'moxfield\.com/decks/([A-Za-z0-9_-]+)', url)
    if not m:
        raise ValueError('Could not parse deck ID from URL')
    return m.group(1)


def _parse_moxfield_cards(deck_json: dict) -> list[dict]:
    """Flatten mainboard + commanders + sideboard into a card list."""
    cards = []
    sections = {
        'commander': deck_json.get('commanders', {}),
        'mainboard': deck_json.get('mainboard', {}),
        'sideboard': deck_json.get('sideboard', {}),
        'companion': deck_json.get('companions', {}),
    }
    for section, entries in sections.items():
        for _key, entry in entries.items():
            card = entry.get('card', {})
            cards.append({
                'name': card.get('name', ''),
                'set_id': card.get('set', '').lower(),
                'collector_no': str(card.get('cn', '')),
                'scryfall_id': card.get('scryfall_id', ''),
                'quantity': entry.get('quantity', 1),
                'foil': int(entry.get('isFoil', False)),
                'etched': int(entry.get('isEtched', False)),
                'section': section,
            })
    return cards


class MoxfieldPreviewRequest(BaseModel):
    url: str


class MoxfieldSaveRequest(BaseModel):
    collection_name: str
    collection_type: str = 'deck'
    set_id: str | None = None
    purchase_date: str | None = None
    cards: list[dict]


@router.post('/moxfield/preview')
def moxfield_preview(req: MoxfieldPreviewRequest):
    """Fetch a Moxfield deck by URL and return a preview (no DB writes)."""
    try:
        deck_id = _extract_deck_id(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    resp = requests.get(_MOXFIELD_DECK_URL.format(deck_id), timeout=15, headers={
        'User-Agent': 'mtg-api/1.0'
    })
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail='Deck not found on Moxfield')
    if not resp.ok:
        raise HTTPException(status_code=502, detail=f'Moxfield API error: {resp.status_code}')

    deck_json = resp.json()
    cards = _parse_moxfield_cards(deck_json)

    return {
        'deck_name': deck_json.get('name', ''),
        'format': deck_json.get('format', ''),
        'card_count': sum(c['quantity'] for c in cards),
        'cards': cards,
    }


@router.post('/moxfield/save')
def moxfield_save(req: MoxfieldSaveRequest):
    """
    SSE stream — saves a Moxfield deck preview to the DB card by card.

    Emits: data: {"done": N, "total": M, "card": "Name", "fetched": bool}
    Final:  data: {"complete": true, "collection_id": N, "imported": M}
    Error:  data: {"error": "message"}
    """
    def event_stream():
        conn = get_conn()
        try:
            existing = conn.execute(
                'SELECT id FROM collections WHERE name = ?', (req.collection_name,)
            ).fetchone()
            if existing:
                yield f'data: {json.dumps({"error": "Collection name already exists"})}\n\n'
                return

            cur = conn.execute(
                'INSERT INTO collections (name, set_id, type) VALUES (?, ?, ?)',
                (req.collection_name, req.set_id, req.collection_type),
            )
            collection_id = cur.lastrowid

            total = len(req.cards)
            imported = 0

            for i, card in enumerate(req.cards):
                scryfall_id = card.get('scryfall_id', '')
                set_id = card.get('set_id', '')
                collector_no = card.get('collector_no', '')
                name = card.get('name', '')
                fetched = False

                existing_card = conn.execute(
                    'SELECT id FROM cards WHERE id = ?', (scryfall_id,)
                ).fetchone() if scryfall_id else None

                if existing_card is None and set_id and collector_no:
                    time.sleep(0.05)
                    card_json = fetch_card(set_id, collector_no)
                    if card_json:
                        scryfall_id = upsert_card(conn, card_json)
                        fetched = True

                if scryfall_id:
                    conn.execute("""
                        INSERT INTO collection_cards
                            (collection_id, card_id, quantity, foil, etched)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT DO NOTHING
                    """, (
                        collection_id,
                        scryfall_id,
                        card.get('quantity', 1),
                        card.get('foil', 0),
                        card.get('etched', 0),
                    ))
                    imported += 1

                yield f'data: {json.dumps({"done": i + 1, "total": total, "card": name, "fetched": fetched})}\n\n'

            conn.commit()
            yield f'data: {json.dumps({"complete": True, "collection_id": collection_id, "imported": imported})}\n\n'

        except Exception as e:
            conn.rollback()
            yield f'data: {json.dumps({"error": str(e)})}\n\n'
        finally:
            conn.close()

    return StreamingResponse(event_stream(), media_type='text/event-stream')
