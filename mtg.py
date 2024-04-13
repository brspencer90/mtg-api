# %%
import requests as req
import time, json, gzip, csv
import pandas as pd
import re
from sklearn.preprocessing import MultiLabelBinarizer

from constants import Constants as c

import itertools
import plotly.graph_objects as go
import numpy as np

# %%
def parse_mtga_export(fn='mtga_export.txt'):
    list_card = open(fn,'r').read().split('\n')[1:]

    quantity = '^\d{1}'
    card_id = '\d{1,4}$'
    card_name = '\d (.*?) \('
    deck = '\((.*?)\)'

    id_r = [
            [
                re.findall(quantity,x)[0],
                re.findall(card_name,x)[0],
                re.findall(deck,x)[0],
                re.findall(card_id,x)[0]
            ] for x in list_card]

    return id_r

def pull_parse_file(source:str = 'custom',set_id='otj'):

    assert source in ['custom','mtga'], "invalid source input, valid inputs are : ['custom', 'mtga']"

    list_set = []

    if source == 'custom':
        fn = 'Deck.txt'
        id_r = open(fn,'r').read().split('\n')

        for id in id_r:
            list_card = get_card_info(id,set_id)
            list_set.append(list_card)

    elif source == 'mtga':
        df = pd.DataFrame(parse_mtga_export(),columns=['qty','name','deck','id'])

        for idx in list(df.index):
            id = df.loc[idx,'id']
            set_id = df.loc[idx,'deck'].lower()

            list_card = get_card_info(id,set_id)
            list_set.append(list_card)

    return pd.DataFrame(columns=c.col,data=list_set)

def get_card_info(id,set_id):
    time.sleep(.1)
    card_json = json.loads(req.get(f'https://api.scryfall.com/cards/{set_id}/{id}').text)

    if 'card_faces' in card_json:
        name = [card['name'] for card in card_json['card_faces']]
        type_line = [card['type_line'] for card in card_json['card_faces']]
        oracle_text = [card['oracle_text'] for card in card_json['card_faces']]
    else:
        name = card_json['name']
        type_line = card_json['type_line']
        oracle_text = card_json['oracle_text']

    # Enocde types
    type_creature = 0
    type_noncreature = 0
    type_land = 0

    if 'Creature' in type_line:
        type_creature = 1
    elif 'Land' in type_line:
        type_land = 1
    else:
        type_noncreature = 1

    # One-hot encode colours
    colours = card_json['colors']

    if len(colours) == 0: # ['B','W','R','G','U']:
        card_json['colors'] = ['N']
        

    if ('power' in card_json) | ('toughness' in card_json):
        power = card_json['power']
        toughness = card_json['toughness']
    else:
        power = ''
        toughness = ''

    price_std = card_json['prices']['usd']
    price_foil = card_json['prices']['usd_foil']
    price_etch = card_json['prices']['usd_etched']

    keywords = card_json['keywords']

    # otj specific : 
    # add keyword for outlaw definition
    if any([True for outlaw in ['Assassin','Mercenary','Warlock','Pirate','Rogue','Warlock'] if outlaw in type_line]):
        keywords.append('Outlaw')

    # clean text from parentheticals, keywords
    clean_text = re.sub(r'\([^)]*\)', '', oracle_text)

    for r in c.on_crime_regex_list:
        clean_text = re.sub(r,'',clean_text)

    # add keyword for cause_crime
    if 'target' in clean_text:
        keywords.append('Do Crime')

    # add keyword for on_crime
    if any([True for x in ['committed a crime','commit a crime'] if x in clean_text]):
        keywords.append('On Crime')

    list_card = [name,card_json['mana_cost'],card_json['cmc'],power,toughness,
                    type_line,type_creature,type_noncreature,type_land,oracle_text,
                    card_json['colors'],card_json['color_identity'],keywords,card_json['rarity'],card_json['collector_number'],
                    price_std,price_foil,price_etch]
    
    list_gf = [card_json['name'],set_id,card_json['set_name'],1,'','']

    return list_card

def encode_features(df):

    list_kw = list(set(df['Keywords'].sum()))
    list_crea_type = list(set(itertools.chain(*[re.sub('Creature — ','',x).split() for x in df[df['Creature'] == 1]['Type']])))

    df['Card Type'] = df[c.list_card_type].idxmax(1)

    mlb = MultiLabelBinarizer()
    # One-hot encode creature type
    df_creature = df[df['Creature'] == 1]
    df_creature['Creature_Type'] = [re.sub('Creature — ','',x).split() for x in df_creature['Type']]
    df = df.join(pd.DataFrame(mlb.fit_transform(df_creature.pop('Creature_Type')),
                            columns=['Creature_'+x for x in list(mlb.classes_)],
                            index=df_creature.index))


    # One-hot encode keywords
    df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('Keywords')),
                            columns=['Key_'+x for x in list(mlb.classes_)],
                            index=df.index))

    # One-hot encode colours
    mlb = MultiLabelBinarizer(classes=c.list_colour_abbr)
    df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('Colours')),
                            columns=[c.dict_colour[x] for x in mlb.classes_],
                            index=df.index))
    df['Colour'] = df[c.list_colour].idxmax(1)

    return df

def visualize_deck(df):
    df_colour = df[c.list_colour].sum()

    # Explore by colour
    fig = go.Figure(
            data=go.Pie(values=df_colour.values,labels=df_colour.index,marker_colors=[c.dict_colour_map[x] for x in df_colour.index])
    )
    fig.update_layout(height = 600, width = 800)
    fig.update(layout_title_text='Card Colours')
    fig.show()

    # Explore card type by colour
    df_colourtype_gb = df.groupby('Card Type')[c.list_colour].sum().T
    df_colourtype_gb_labels = df_colourtype_gb.index

    fig = go.Figure(
        [go.Bar(name=x,x=df_colourtype_gb_labels,y=df_colourtype_gb[x]) for x in c.list_card_type[:2]]
    )
    fig.update_layout(height = 600, width = 800,barmode='stack')
    fig.update(layout_title_text='Card Type by Colours')
    fig.show()

    # Explore rarity by colour
    df_colourtype_gb = df.groupby('Rarity')[c.list_colour].sum().T
    df_colourtype_gb_labels = df_colourtype_gb.index

    fig = go.Figure(
        [go.Bar(name=x,x=df_colourtype_gb_labels,y=df_colourtype_gb[x]) for x in c.list_rarity]
    )
    fig.update_layout(height = 600, width = 800,barmode='stack')
    fig.update(layout_title_text='Rarity by Colours')
    fig.show()


    # Explore mana curve by colour
    df_colourmana_creatures_gb = df[df['Card Type'] == 'Creature'].groupby('CMC')[c.list_colour].sum().T
    df_colourmana_noncreatures_gb = df[df['Card Type'] == 'Non-Creature'].groupby('CMC')[c.list_colour].sum().T
    list_cmc = list(np.arange(0.,df['CMC'].max()))

    for color in c.list_colour[:-1]:
        df_creatures = df_colourmana_creatures_gb.loc[color,:]
        df_noncreatures = df_colourmana_creatures_gb.loc[color,:]

        fig = go.Figure(
            [go.Bar(name='Creatures',x=df_creatures.index,y=df_creatures.values,marker_color='slategrey'),
            go.Bar(name='Non-Creatures',x=df_noncreatures,y=df_noncreatures.values,marker_color='crimson')]
        )
        fig.update_layout(height = 600, width = 1000,barmode='stack')
        fig.update(layout_title_text=f'Mana Curve by Colour : {color}')
        fig.show()

# %%
def get_all_from_set(set_id):
    list_set = []
    try:
        for id in range(1,500):
            list_card = get_card_info(id,set_id)
            list_set.append(list_card)
            df = pd.DataFrame(columns=c.col,data=list_set)
    except:
        return encode_features(df)
    
def get_scores(df,set_id):
    df_scores = pd.read_csv('otj_pre-release_scores_lr.csv')
    df_merge = pd.merge(df,df_scores[['Card name','Score by Marshall','Score by Luis']],how='left',left_on='Name',right_on=['Card name'])

    df['Score Combined'] = df_merge['Score by Marshall'].str.split('/')+ (df_merge['Score by Luis'].str.split('/'))

    for idx in df_merge['Score Combined'].dropna().index.to_list():
        df.loc[idx,'GPA Average'] = np.mean([c.dict_scores[x] for x in df.loc[idx,'Score Combined']])

# %%
