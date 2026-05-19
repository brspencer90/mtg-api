"""Helpers that convert Plotly figures to JSON for API responses."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from constants import Constants as c
from mtg_plot import plot_set_specific, plot_simple_bar


def _safe_json(fig) -> str:
    return fig.to_json() if fig is not None else '{}'


def build_color_pie(df: pd.DataFrame) -> str:
    df_colour = df[c.list_colour].sum()
    fig = go.Figure(data=go.Pie(
        values=df_colour.values,
        labels=df_colour.index,
        marker_colors=[c.dict_colour_map[x] for x in df_colour.index],
    ))
    fig.update_layout(height=600, width=800, template='plotly_dark')
    fig.update(layout_title_text='Card Colours')
    return _safe_json(fig)


def build_mana_pip_pie(df: pd.DataFrame) -> str:
    df_mana = df[c.list_mana_colour].sum()
    fig = go.Figure(data=go.Pie(
        values=df_mana.values,
        labels=[x.split('_')[1] for x in df_mana.index],
        marker_colors=[c.dict_colour_map[x.split('_')[1]] for x in df_mana.index],
    ))
    fig.update_layout(height=600, width=800, template='plotly_dark')
    fig.update(layout_title_text='Mana Pips')
    return _safe_json(fig)


def build_type_by_colour(df: pd.DataFrame) -> str:
    fig = plot_simple_bar(df, x_axes=c.list_colour, y_col_name='Card Type',
                           y_stack=c.list_card_type[:2], title='Card Type by Colours')
    return _safe_json(fig)


def build_rarity_by_colour(df: pd.DataFrame) -> str:
    fig = plot_simple_bar(df, x_axes=c.list_colour, y_col_name='Rarity',
                           y_stack=c.list_rarity, title='Rarity by Colours')
    return _safe_json(fig)


def build_mana_curve(df: pd.DataFrame) -> str:
    df_creatures = df[df['Card Type'] == 'Creature'].groupby('CMC')[c.list_colour].sum().T
    df_noncreatures = df[df['Card Type'] == 'Non-Creature'].groupby('CMC')[c.list_colour].sum().T
    list_cmc = list(np.arange(1, df['CMC'].max() + 1))

    c_sum = df_creatures.sum().reindex(list_cmc, fill_value=0)
    nc_sum = df_noncreatures.sum().reindex(list_cmc, fill_value=0)

    fig = go.Figure([
        go.Bar(name='Creatures', x=list_cmc, y=c_sum.values, marker_color='slategrey'),
        go.Bar(name='Non-Creatures', x=list_cmc, y=nc_sum.values, marker_color='crimson'),
    ])
    fig.update_layout(height=600, width=1000, barmode='stack', template='plotly_dark')
    fig.update(layout_title_text='Overall Mana Curve')
    return _safe_json(fig)


def build_set_specific_charts(df: pd.DataFrame, set_id: str) -> list[str]:
    figs = plot_set_specific(df, set_id) or []
    return [_safe_json(f) for f in figs]


def build_all_charts(df: pd.DataFrame, set_id: str) -> dict[str, str]:
    return {
        'color_pie': build_color_pie(df),
        'mana_pip_pie': build_mana_pip_pie(df),
        'type_by_colour': build_type_by_colour(df),
        'rarity_by_colour': build_rarity_by_colour(df),
        'mana_curve': build_mana_curve(df),
        'set_specific': json.dumps(build_set_specific_charts(df, set_id)),
    }
