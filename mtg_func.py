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

from mtg_keywords import otj_keywords, mh3_keywords
 
# %%

def get_list_of_sets():
    from datetime import datetime as dt 
    set_json = json.loads(req.get(f'https://api.scryfall.com/sets').text)
    
    return [x['scryfall_uri'].split('/sets/')[1] for x in set_json['data'] if ((dt.strptime(x['released_at'],'%Y-%m-%d') < dt.now()) & (x['set_type'] == 'expansion'))]

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

def pull_parse_file(source:str = 'deck',set_id='otj'):

    if source == 'deck':
        fn = 'Deck.txt'
        id_r = open(fn,'r').read().split('\n')

        list_set = loop_cards(id_r,set_id)
    elif source == 'mtga':
        df = pd.DataFrame(parse_mtga_export(),columns=['qty','name','deck','id'])

        for idx in list(df.index):
            id = df.loc[idx,'id']
            set_id = df.loc[idx,'deck'].lower()

            list_card = get_card_info(id,set_id)
            list_set.append(list_card)
    else:
        fn = source
        id_r = open(fn,'r').read().split('\n')

        list_set = loop_cards(id_r,set_id)

    return pd.DataFrame(columns=c.col,data=list_set)

def get_card_info(id,set_id,foil=False,etch=False):
    time.sleep(.1)
    card_json = json.loads(req.get(f'https://api.scryfall.com/cards/{set_id}/{id}').text)

    if card_json['object'] != 'error':
        
        if 'card_faces' in card_json:
            name = [card['name'] for card in card_json['card_faces']]
            type_line = [card['type_line'] for card in card_json['card_faces']]
            oracle_text = [card['oracle_text'] for card in card_json['card_faces']]

            keywords = card_json['keywords']

            # clean text from parentheticals, keywords
            clean_text = [re.sub(r'\([^)]*\)', '', text) for text in oracle_text]

            # add otj specific keywords
            keywords = otj_keywords(keywords,type_line,clean_text,double_face=True)

            # add mh3 keywords
            keywords = mh3_keywords(keywords,type_line,clean_text,double_face=True)

        else:
            name = card_json['name']
            type_line = card_json['type_line']
            oracle_text = card_json['oracle_text']

            keywords = card_json['keywords']

            # clean text from parentheticals, keywords
            clean_text = re.sub(r'\([^)]*\)', '', oracle_text)

            # add otj specific keywords
            keywords = otj_keywords(keywords,type_line,clean_text,double_face=False)

            # add mh3 keywords
            keywords = mh3_keywords(keywords,type_line,clean_text,double_face=False)

        # Enocde types
        type_creature = 0
        type_noncreature = 0
        type_land = 0
        type_plane = 0
        type_instant = 0
        type_sorcery = 0
        type_enchantment = 0
        type_artifact = 0

        if 'Creature' in type_line:
            type_creature = 1
        elif 'Land' in type_line:
            type_land = 1
        elif 'Planeswalker' in type_line:
            type_plane = 1
        else:
            type_noncreature = 1
        
        if 'Instant' in type_line:
            type_instant = 1
        if 'Sorcery' in type_line:
            type_sorcery = 1
        if 'Enchantment' in type_line:
            type_enchantment = 1
        if 'Artifact' in type_line:
            type_artifact = 1

        # Encode colours
        if 'colors' not in card_json.keys():
            colours = list(set([x for xs in [card['colors'] for card in card_json['card_faces']] for x in xs]))
        else:
            colours = card_json['colors']

        if len(colours) == 0: # ['B','W','R','G','U']:
            colours = ['N']            

        # Extract mana cost
        if 'mana_cost' in card_json.keys():
            mana_cost = card_json['mana_cost']
        else:
            mana_cost = [card['mana_cost'] for card in card_json['card_faces'] if len(card['mana_cost'])>0]

            mana_cost = mana_cost[0] if len(mana_cost) == 1 else mana_cost

        # Extract power / toughness
        if ('power' in card_json) | ('toughness' in card_json):
            power = card_json['power']
            toughness = card_json['toughness']
        else:
            power = ''
            toughness = ''

        # Extract prices
        price_std = float(card_json['prices']['usd']) if bool(card_json['prices']['usd']) else None
        price_foil = float(card_json['prices']['usd_foil']) if bool(card_json['prices']['usd_foil']) else None
        price_etch = float(card_json['prices']['usd_etched']) if bool(card_json['prices']['usd_etched']) else None

        if foil : 
            price = price_foil
        elif etch : 
            price = price_etch
        else: 
            price = price_std

        price = price if price else 0

        list_card = [name,mana_cost,card_json['cmc'],power,toughness,
                        type_line,type_creature,type_noncreature,type_plane,type_land,type_instant,type_sorcery,type_enchantment,type_artifact,oracle_text,
                        colours,card_json['color_identity'],keywords,card_json['rarity'],card_json['collector_number'],
                        price,price_std,price_foil,price_etch]
        
        list_gf = [card_json['name'],set_id,card_json['set_name'],1,'','']

        return list_card
    
    else:
        return card_json

def encode_features(df):

    list_kw = list(set(df['Keywords'].sum()))
    list_crea_type = list(set(itertools.chain(*[re.sub('Creature — ','',x).split() for x in df[df['Creature'] == 1]['Type']])))

    df['Card Type'] = df[c.list_card_type].idxmax(1)

    mlb = MultiLabelBinarizer()
    # One-hot encode creature type
    df_creature = df[df['Creature'] == 1].copy()
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
    
    # if sum c.list_colour > 1 -> gold

    df['Colour'] = df[c.list_colour].idxmax(1)

    # Encode colours (count)
    for colour in c.list_colour:
        df[f'Mana_{colour}'] = df['Mana Cost'].apply(lambda col: col.count(colour))

    return df

def loop_cards(id_r,set_id):
    list_set = []
    
    for id in id_r:

        if 'f' in id:
            foil = True
            etch = False
            id = id[:-1]
        elif 'e' in id:
            foil = False
            etch = True
            id = id[:-1]
        else:
            foil = False
            etch = False
        
        list_card = get_card_info(id,set_id,foil,etch)
        list_set.append(list_card)

    return list_set

# %%
def get_scores(df,set_id):
    df_scores = pd.read_csv('otj_pre-release_scores_lr.csv')
    df_merge = pd.merge(df,df_scores[['Card name','Score by Marshall','Score by Luis']],how='left',left_on='Name',right_on=['Card name'])

    df['Score Combined'] = df_merge['Score by Marshall'].str.split('/')+ (df_merge['Score by Luis'].str.split('/'))

    for idx in df['Score Combined'].dropna().index.to_list():
        df.loc[idx,'GPA Average'] = np.mean([c.dict_scores[x] for x in df.loc[idx,'Score Combined']])

def parse_dm_file(file_name):
    df_id = pd.read_csv(file_name,sep=' ',header=None)
    list_id = []
    for idx in df_id.index.to_list():
        list_id += [int(df_id.loc[idx,1])]*int(df_id.loc[idx,0])

    with open("output.txt", "w") as file:
        for item in list_id:
            file.write(f"{item}\n")