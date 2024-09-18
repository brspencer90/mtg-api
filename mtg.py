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

from mtg_plot import blb_plots, mh3_plots, otj_plots, plot_simple_bar
from mtg_func import get_card_info, encode_features

# %%
def get_all_from_set(set_id):
    set_json = json.loads(req.get(f'https://api.scryfall.com/sets/{set_id}').text)
    list_set = []

    cons_errors = 0
    card_count = 0
    id = 1

    while ((card_count < set_json['card_count']) & (cons_errors <= 10)):
        list_card = get_card_info(id,set_id)
        
        if type(list_card) == list:

            list_set.append(list_card)
            df = pd.DataFrame(columns=c.col,data=list_set)
            card_count += 1
            cons_errors = 0
        elif type(list_card) == dict:
            print(f'no values for id no : {id}')
            cons_errors += 1

        id += 1

    df = df.drop_duplicates(subset='Name')

    return encode_features(df)

def visualize_deck(df,set_id):

    # Explore by colour
    df_colour = df[c.list_colour].sum()

    fig = go.Figure(
            data=go.Pie(values=df_colour.values,labels=df_colour.index,marker_colors=[c.dict_colour_map[x] for x in df_colour.index])
    )
    fig.update_layout(height = 600, width = 800)
    fig.update(layout_title_text='Card Colours')
    fig.show()

    # Explore card type by colour
    plot_simple_bar(
        df,
        x_axes = c.list_colour,
        y_col_name = 'Card Type',
        y_stack = c.list_card_type[:2],
        title = 'Card Type by Colours'
    )

    # Explore rarity by colour
    plot_simple_bar(
        df,
        x_axes = c.list_colour,
        y_col_name = 'Rarity',
        y_stack = c.list_rarity,
        title = 'Rarity by Colours'
    )

    # Explore rarity by card type
    plot_simple_bar(
        df,
        x_axes = c.list_card_type,
        y_col_name = 'Rarity',
        y_stack = c.list_rarity,
        title = 'Rarity by Card Type'
    )

    if set_id == 'otj':
        otj_plots(df)
    elif set_id == 'mh3':
        mh3_plots(df)
    elif set_id == 'blb':
        blb_plots(df)

    # Explore mana curve by colour
    df_colourmana_creatures_gb = df[df['Card Type'] == 'Creature'].groupby('CMC')[c.list_colour].sum().T
    df_colourmana_noncreatures_gb = df[df['Card Type'] == 'Non-Creature'].groupby('CMC')[c.list_colour].sum().T
    list_cmc = list(np.arange(1,df['CMC'].max()+1))

    for color in c.list_colour:
        df_creatures = df_colourmana_creatures_gb.loc[color,:].reindex(list_cmc,fill_value=0)
        df_noncreatures = df_colourmana_noncreatures_gb.loc[color,:].reindex(list_cmc,fill_value=0)

        fig = go.Figure(
            [go.Bar(name='Creatures',x=list_cmc,y=df_creatures.values,marker_color='slategrey'),
            go.Bar(name='Non-Creatures',x=list_cmc,y=df_noncreatures.values,marker_color='crimson')]
        )

        fig.update_layout(height = 600, width = 1000,barmode='stack')
        fig.update(layout_title_text=f'Mana Curve by Colour : {color}')
        fig.show()

    # Explore combined mana curve
    df_colourmana_creatures_gb = df[df['Card Type'] == 'Creature'].groupby('CMC')[c.list_colour].sum().T
    df_colourmana_noncreatures_gb = df[df['Card Type'] == 'Non-Creature'].groupby('CMC')[c.list_colour].sum().T
    list_cmc = list(np.arange(1,df['CMC'].max()+1))

    df_creatures = df_colourmana_creatures_gb.sum().reindex(list_cmc,fill_value=0)
    df_noncreatures = df_colourmana_noncreatures_gb.sum().reindex(list_cmc,fill_value=0)

    fig = go.Figure(
        [go.Bar(name='Creatures',x=list_cmc,y=df_creatures.values,marker_color='slategrey'),
        go.Bar(name='Non-Creatures',x=list_cmc,y=df_noncreatures.values,marker_color='crimson')]
    )

    fig.update_layout(height = 600, width = 1000,barmode='stack')
    fig.update(layout_title_text=f'Overall Mana Curve by Colour')
    fig.show()

def get_word_frequency(set_list):
    if isinstance(set_list,str):
        set_list = [set_list]

    word_list = [] 

    for set_id in set_list:
        print(f'start {set_id}')
        df = get_all_from_set(set_id)
    
        text = df['Text'].str.lower().replace(r'\n',' ',regex=True).str.cat(sep=' ')
        text_no_paranth = re.sub(r'\([^)]*\)', '', text)
        text_no_bracket = re.sub(r'\{[^}]*}', '', text_no_paranth)
        words = nltk.tokenize.word_tokenize(text_no_bracket)
        words_no_punc = [x for x in words if re.compile(r'\w+').match(x)]

        stopwords = nltk.corpus.stopwords.words('english')
        word_list_sub = [w for w in words_no_punc if w not in stopwords]

        word_list.extend(word_list_sub)    

    word_list_dist = nltk.FreqDist(word_list)

    rslt = pd.DataFrame.from_dict(
                        data=dict(word_list_dist),
                        columns=['Count'],
                        orient='index'
                    )

    return rslt.sort_values('Count',ascending=False)

# %%


# To Do : 
    # how to visualize deck if no creature types are available (Eldrazi)
    # rarity by card type
    # include lands / card type
    # mana pings for land distribution
    # card type for dual faced cards

# blb : 