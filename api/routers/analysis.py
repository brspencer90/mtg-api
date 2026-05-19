"""
Analysis router — loads cards from the DB, encodes features, runs visualize_deck,
and returns Plotly figure dicts the frontend renders with react-plotly.js.
"""
import json

import pandas as pd
from fastapi import APIRouter, HTTPException

from constants import Constants as c
from db.connection import get_conn
from mtg_func import encode_features
from mtg import visualize_deck

router = APIRouter(prefix='/api', tags=['analysis'])


def _load_collection_df(collection_id: int) -> tuple[pd.DataFrame, str]:
    """Load a collection from the DB into an encoded DataFrame. Returns (df, set_id)."""
    conn = get_conn()
    try:
        coll = conn.execute(
            'SELECT * FROM collections WHERE id = ?', (collection_id,)
        ).fetchone()
        if coll is None:
            raise ValueError('Collection not found')

        rows = conn.execute("""
            SELECT c.*, cc.quantity, cc.foil, cc.etched
            FROM collection_cards cc
            JOIN cards c ON c.id = cc.card_id
            WHERE cc.collection_id = ?
        """, (collection_id,)).fetchall()

        if not rows:
            raise ValueError('Collection has no cards')

        records = []
        for r in rows:
            rec = dict(r)
            # Deserialise JSON columns back to lists (encode_features expects lists)
            rec['colours'] = json.loads(rec.get('colours') or '[]')
            rec['colour_identity'] = json.loads(rec.get('colour_identity') or '[]')
            rec['keywords'] = json.loads(rec.get('keywords') or '[]')
            records.append(rec)

        df = pd.DataFrame(records)

        # Map DB column names → Constants.col names expected by encode_features
        rename = {
            'name': 'Name',
            'mana_cost': 'Mana Cost',
            'cmc': 'CMC',
            'power': 'Power',
            'toughness': 'Toughness',
            'type_line': 'Type',
            'oracle_text': 'Text',
            'colours': 'Colours',
            'colour_identity': 'Colour Identity',
            'keywords': 'Keywords',
            'rarity': 'Rarity',
            'collector_no': 'Collector No.',
            'set_id': 'Set',
            'price': 'Price',
            'price_std': 'Price STD',
            'price_foil': 'Price Foil',
            'price_etched': 'Price Etched',
        }
        df = df.rename(columns=rename)

        # Add missing type flag columns (encode_features reads these)
        for col in ['Creature', 'Non-Creature', 'Planeswalker', 'Land',
                    'Instant', 'Sorcery', 'Enchantment', 'Artifact']:
            if col not in df.columns:
                df[col] = df['Type'].apply(lambda t: int(col in str(t))) if 'Type' in df.columns else 0

        df = encode_features(df)
        return df, coll['set_id'] or ''
    finally:
        conn.close()


@router.get('/collections/{collection_id}/analysis')
def analyse_collection(collection_id: int):
    try:
        df, set_id = _load_collection_df(collection_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    figs = visualize_deck(df, set_id)
    return {'figures': [fig.to_dict() for fig in figs]}
