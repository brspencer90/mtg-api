"""Sealed/draft pool analysis — wraps existing mtg_func pipeline."""
import uuid

import pandas as pd

from mtg_func import encode_features, get_card_info, loop_cards, parse_mtga_export
from constants import Constants as c
from api.services.plot_service import build_all_charts
from api.models.responses import PoolAnalysisResponse, PoolSummary


def analyze_pool_from_text(raw_text: str, set_id: str | None = None) -> PoolAnalysisResponse:
    """Accept MTGA export text and return full pool analysis."""
    parsed = parse_mtga_export(raw_text=raw_text)

    list_set = []
    for row in parsed:
        _qty, _name, card_set, card_id = row
        resolved_set = set_id or card_set.lower()
        card = get_card_info(card_id, resolved_set)
        if isinstance(card, list):
            list_set.append(card)

    if not list_set:
        raise ValueError('No cards could be fetched from the provided export.')

    df = pd.DataFrame(columns=c.col, data=list_set)
    df = df.drop_duplicates(subset='Name')
    df = encode_features(df)

    resolved_set_id = set_id or parsed[0][2].lower()
    charts = build_all_charts(df, resolved_set_id)
    summary = _build_summary(df)

    return PoolAnalysisResponse(
        pool_id=str(uuid.uuid4()),
        cards=df.reset_index(drop=True).to_dict(orient='records'),
        charts=charts,
        summary=summary,
    )


def analyze_pool_from_ids(id_list: list[str], set_id: str) -> PoolAnalysisResponse:
    """Accept list of collector numbers (with optional f/e suffix) and set_id."""
    list_set = loop_cards(id_list, set_id)
    list_set = [c for c in list_set if isinstance(c, list)]

    if not list_set:
        raise ValueError('No cards could be fetched.')

    df = pd.DataFrame(columns=c.col, data=list_set)
    df = df.drop_duplicates(subset='Name')
    df = encode_features(df)

    charts = build_all_charts(df, set_id)
    summary = _build_summary(df)

    return PoolAnalysisResponse(
        pool_id=str(uuid.uuid4()),
        cards=df.reset_index(drop=True).to_dict(orient='records'),
        charts=charts,
        summary=summary,
    )


def _build_summary(df: pd.DataFrame) -> PoolSummary:
    color_counts = {col: int(df[col].sum()) for col in c.list_colour if col in df.columns}
    avg_cmc = float(df[df['Card Type'] != 'Land']['CMC'].mean()) if len(df) > 0 else 0.0

    key_cols = [col for col in df.columns if col.startswith('Key_')]
    keyword_counts = {col.replace('Key_', ''): int(df[col].sum()) for col in key_cols if df[col].sum() > 0}

    return PoolSummary(
        total_cards=len(df),
        color_counts=color_counts,
        avg_cmc=round(avg_cmc, 2),
        keyword_counts=keyword_counts,
    )
