# %%
import requests as req
import time, json, gzip, csv
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer

from constants import Constants as c

import warnings
warnings.simplefilter(action='ignore')

import itertools
import plotly.graph_objects as go
import numpy as np
import nltk

# %%
def plot_set_specific(df,set_id):
    if set_id == 'otj':
        otj_plots(df)
    elif set_id == 'mh3':
        mh3_plots(df)
    elif set_id == 'blb':
        blb_plots(df)
    elif set_id == 'dsk':
        plot_simple_bar(
            df,
            x_axes = c.dsk_archetype_keys,
            y_col_name = 'Colour',
            y_stack = c.list_colour,
            title = 'Archetype by Colours'
        )
    

def plot_simple_bar(df:pd.DataFrame,x_axes:list,y_col_name:str,y_stack:list=None,title:str=''):
    list_x = x_axes

    list_not = [x for x in list_x if x not in df.columns]
    for col in list_not:
        df[col] = 0

    df_values = df.groupby(y_col_name)[list_x].sum().T

    if bool(y_stack) & (y_stack == c.list_colour):
        fig = go.Figure(
            [go.Bar(name=x,x=list_x,y=df_values[x],marker_color=c.dict_colour_map[x]) for x in y_stack]
        )
    elif y_stack:
        fig = go.Figure(
            [go.Bar(name=x,x=list_x,y=df_values[x]) for x in y_stack]
        )
    else:
        fig = go.Figure(
            go.Bar(x=list_x,y=df_values)
        )

    fig.update_layout(
                height = 600, 
                width = 800,
                barmode='stack',
                template='plotly_dark'
            )
    
    fig.update(layout_title_text=title)
    
    return fig.show()

def blb_plots(df:pd.DataFrame):
    plot_simple_bar(
        df,
        x_axes = c.blb_archetype_creatures,
        y_col_name = 'Colour',
        y_stack = c.list_colour,
        title = 'Archetype by Colours'
    )

    plot_simple_bar(
        df,
        x_axes = c.blb_archetype_keys,
        y_col_name = 'Colour',
        y_stack = c.list_colour,
        title = 'Archetype Keywords by Colours'
    )

def otj_plots(df):
    # Explore by Outlaw
    df_outlaw_gb = df[df['Key_Outlaw'] == 1][c.list_colour].sum().T
    df_outlaw_gb_labels = df_outlaw_gb.index

    fig = go.Figure(
        go.Bar(x=df_outlaw_gb_labels,y=df_outlaw_gb)
    )
    fig.update_layout(height = 600, width = 800,barmode='stack')
    fig.update(layout_title_text='Outlaw by Colours')
    fig.show()

    # Explore Crime by Colours
    df_on_crime_gb = df[df['Key_On Crime'] == 1][c.list_colour].sum().T
    df_do_crime_gb = df[df['Key_Do Crime'] == 1][c.list_colour].sum().T
    df_crime_labels = c.list_colour 

    fig = go.Figure(
        [
            go.Bar(name='Do Crime',x=df_crime_labels,y=df_do_crime_gb),
            go.Bar(name='On Crime',x=df_crime_labels,y=df_on_crime_gb)
        ]
    )
    fig.update_layout(height = 600, width = 800,barmode='group')
    fig.update(layout_title_text='Crimeness by Colours')

    fig.show()

    # Explore Plot by Creatures & Colours
    df_plot_gb = df[df['Key_Plot'] == 1][c.list_colour].sum().T
    df_plot_creatures = df[(df['Key_Plot']== 1) & (df['Card Type'] == 'Creature')][c.list_colour].sum().T
    df_plot_noncreatures = df[(df['Key_Plot']== 1) & (df['Card Type'] == 'Non-Creature')][c.list_colour].sum().T

    df_plot_labels = c.list_colour 

    fig = go.Figure(
        [
            go.Bar(name='Creatures Plot',x=df_plot_labels,y=df_plot_creatures),
            go.Bar(name='Non-Creatures Plot',x=df_plot_labels,y=df_plot_noncreatures),
        ]
    )
    fig.update_layout(height = 600, width = 800,barmode='group')
    fig.update(layout_title_text='Plot by Colours')

    fig.show()

def mh3_plots(df):
    # Explore Archetypes by Count
    list_arche = ['Key_Energy','Key_Modified','Key_Eldrazi','Key_Artifacts']
    df_arche_labels = [label.split('Key_')[1] for label in ['Key_Energy','Key_Modified','Key_Eldrazi','Key_Artifacts']]
    
    mask = (df['Key_Energy'] == 1) | (df['Key_Modified'] == 1) | (df['Key_Eldrazi'] == 1) | (df['Key_Artifacts'] == 1) 
    df_plot_gb = df[mask][list_arche].sum().T
    df_plot_creatures = df[mask & (df['Card Type'] == 'Creature')][list_arche].sum().T
    df_plot_noncreatures = df[mask & (df['Card Type'] == 'Non-Creature')][list_arche].sum().T
    
    fig = go.Figure(
        [
            go.Bar(name='Creatures Archetype',x=df_arche_labels,y=df_plot_creatures),
            go.Bar(name='Non-Creatures Archetype',x=df_arche_labels,y=df_plot_noncreatures),
        ]
    )
    fig.update_layout(height = 600, width = 800,barmode='group')
    fig.update(layout_title_text='Count of Archetype Cards')

    fig.show()

def plot_cp_manacurve(df,set_id,cp:str='WR'):
    color_pair_list = [c.dict_colour[color] for color in cp]
    cp_title = str(color_pair_list).replace("'",'')[1:-1]

    df_cp = df[pd.DataFrame(data=[df[color] == 1 for color in color_pair_list]).any()]

    plot_set_specific(df_cp,set_id)

    # Explore combined mana curve
    df_colourmana_creatures_gb = df[df['Card Type'] == 'Creature'].groupby('CMC')[color_pair_list].sum().T
    df_colourmana_noncreatures_gb = df[df['Card Type'] == 'Non-Creature'].groupby('CMC')[color_pair_list].sum().T
    list_cmc = list(np.arange(1,df['CMC'].max()+1))

    df_creatures = df_colourmana_creatures_gb.sum().reindex(list_cmc,fill_value=0)
    df_noncreatures = df_colourmana_noncreatures_gb.sum().reindex(list_cmc,fill_value=0)

    fig = go.Figure(
        [go.Bar(name='Creatures',x=list_cmc,y=df_creatures.values,marker_color='slategrey'),
        go.Bar(name='Non-Creatures',x=list_cmc,y=df_noncreatures.values,marker_color='crimson')]
    )

    fig.update_layout(height = 600, width = 1000,barmode='stack',template='plotly_dark')
    fig.update(layout_title_text=f'Overall Mana Curve for {cp_title}')
    fig.show()