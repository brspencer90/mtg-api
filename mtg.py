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

        if 'Creature' in type_line:
            type_creature = 1
        elif 'Land' in type_line:
            type_land = 1
        else:
            type_noncreature = 1

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
                        type_line,type_creature,type_noncreature,type_land,oracle_text,
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

def otj_keywords(keywords,type_line,clean_text,double_face=False):
    # otj specific : 
    if double_face:
        # add keyword for outlaw definition
        if any([True for outlaw in ['Assassin','Mercenary','Warlock','Pirate','Rogue'] if outlaw in type_line]):
            keywords.append('Outlaw')

        for r in c.on_crime_regex_list:
            clean_text = [re.sub(r,'',text) for text in clean_text]

        # add keyword for cause_crime
        if any([True for text in clean_text if 'target' in text]):
            keywords.append('Do Crime')

        # add keyword for on_crime
        if any([[True for key in ['committed a crime','commit a crime'] if key in text] for text in clean_text]):
            keywords.append('On Crime')
    else:
        # add keyword for outlaw definition
        if any([True for outlaw in ['Assassin','Mercenary','Warlock','Pirate','Rogue'] if outlaw in type_line]):
            keywords.append('Outlaw')

        for r in c.on_crime_regex_list:
            clean_text = re.sub(r,'',clean_text)

        # add keyword for cause_crime
        if 'target' in clean_text:
            keywords.append('Do Crime')

        # add keyword for on_crime
        if any([True for x in ['committed a crime','commit a crime'] if x in clean_text]):
            keywords.append('On Crime')

    return keywords

def mh3_keywords(keywords,type_line,clean_text,double_face=False):
    if double_face:
        # Energy Archetypes
        if any([True for text in clean_text if '{E}' in text]):
            keywords.append('Energy')

        if any([True for text in clean_text if re.search(r'[Pp]ay.*{[Ee]}',text)]):
            keywords.append('Use Energy')

        if any([True for text in clean_text if re.search(r'[Gg]et.*{[Ee]}',text)]):
            keywords.append('Get Energy')

        # Eldrazi Archetypes
        if any(True for text in clean_text if (('Eldrazi' in type_line) | ('Devoid' in keywords))):
            keywords.append('Eldrazi')

                # Modified Archetype
        if (
                (('Enchantment Creature','Enchantment - Aura') in type_line) \
                | any([True for key in ['Bestow','Landfall','Outlast','Mentor','Adapt','Modular','Reinforce','Proliferate'] if key in keywords]) \
                | any([bool(re.search(r'[\+\-]\d\/[\+\-]\d counter',text)) for text in clean_text]) \
                | any([bool(re.search(r'[Mm]odified',text)) for text in clean_text]) \
                | any([bool(re.search(r'(?<!energy)(?<!oil) counter',text)) for text in clean_text])
            ):
            keywords.append('Modified')

        # Artifact Archetype
        if (
                (any([True for key in ['Artifact','Equipment'] if key in type_line])) \
                | ('Affinity for artifacts' in clean_text)
            ):
            keywords.append('Artifacts')
    else:
        # Energy Archetypes
        if '{E}' in clean_text:
            keywords.append('Energy')

        if re.search(r'[Pp]ay.*{[Ee]}',clean_text):
            keywords.append('Use Energy')

        if re.search(r'[Gg]et.*{[Ee]}',clean_text):
            keywords.append('Get Energy')

        # Eldrazi Archetypes
        if (('Eldrazi' in type_line) | ('Devoid' in keywords)):
            keywords.append('Eldrazi')

        # Modified Archetype
        if (
                any(True for key in ['Enchantment Creature','Enchantment - Aura'] if key in type_line) \
                | any([True for key in ['Bestow','Landfall','Outlast','Mentor','Adapt','Modular','Reinforce'] if key in keywords]) \
                | bool(re.search(r'[\+\-]\d\/[\+\-]\d counter',clean_text)) \
                | bool(re.search(r'[Mm]odified',clean_text)) \
                | bool(re.search(r'(?<!energy)(?<!oil) counter',clean_text))
            ):
            keywords.append('Modified')

        # Artifact Archetype
        if (
                (any([True for key in ['Artifact','Equipment'] if key in type_line])) \
                | ('Affinity for artifacts' in clean_text)
            ):
            keywords.append('Artifacts')

    return keywords

def visualize_deck(df):

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

    # # Explore by Outlaw
    # df_outlaw_gb = df[df['Key_Outlaw'] == 1][c.list_colour].sum().T
    # df_outlaw_gb_labels = df_outlaw_gb.index

    # fig = go.Figure(
    #     go.Bar(x=df_outlaw_gb_labels,y=df_outlaw_gb)
    # )
    # fig.update_layout(height = 600, width = 800,barmode='stack')
    # fig.update(layout_title_text='Outlaw by Colours')
    # fig.show()

    # # Explore Crime by Colours
    # df_do_crime_gb = df[df['Key_Do Crime'] == 1][c.list_colour].sum().T
    # df_crime_labels = c.list_colour 

    # fig = go.Figure(
    #     [
    #         go.Bar(name='Do Crime',x=df_crime_labels,y=df_do_crime_gb),
    #         go.Bar(name='On Crime',x=df_crime_labels,y=df_on_crime_gb)
    #     ]
    # )
    # fig.update_layout(height = 600, width = 800,barmode='group')
    # fig.update(layout_title_text='Crimeness by Colours')

    # fig.show()

    # # Explore Plot by Creatures & Colours
    # df_plot_gb = df[df['Key_Plot'] == 1][c.list_colour].sum().T
    # df_plot_creatures = df[(df['Key_Plot']== 1) & (df['Card Type'] == 'Creature')][c.list_colour].sum().T
    # df_plot_noncreatures = df[(df['Key_Plot']== 1) & (df['Card Type'] == 'Non-Creature')][c.list_colour].sum().T

    # df_plot_labels = c.list_colour 

    # fig = go.Figure(
    #     [
    #         go.Bar(name='Creatures Plot',x=df_plot_labels,y=df_plot_creatures),
    #         go.Bar(name='Non-Creatures Plot',x=df_plot_labels,y=df_plot_noncreatures),
    #     ]
    # )
    # fig.update_layout(height = 600, width = 800,barmode='group')
    # fig.update(layout_title_text='Plot by Colours')

    # fig.show()

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

def plot_simple_bar(df:pd.DataFrame,x_axes:list,y_col_name:str,y_stack:list=None,title:str=''):
    list_x = x_axes
    df_values = df.groupby(y_col_name)[list_x].sum().T

    if y_stack:
        fig = go.Figure(
            [go.Bar(name=x,x=list_x,y=df_values[x]) for x in y_stack]
        )
    else:
        fig = go.Figure(
            go.Bar(x=list_x,y=df_values)
        )

    fig.update_layout(height = 600, width = 800,barmode='stack')
    fig.update(layout_title_text=title)
    
    return fig.show()

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
    
def get_scores(df,set_id):
    df_scores = pd.read_csv('otj_pre-release_scores_lr.csv')
    df_merge = pd.merge(df,df_scores[['Card name','Score by Marshall','Score by Luis']],how='left',left_on='Name',right_on=['Card name'])

    df['Score Combined'] = df_merge['Score by Marshall'].str.split('/')+ (df_merge['Score by Luis'].str.split('/'))

    for idx in df['Score Combined'].dropna().index.to_list():
        df.loc[idx,'GPA Average'] = np.mean([c.dict_scores[x] for x in df.loc[idx,'Score Combined']])

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

def get_list_of_sets():
    from datetime import datetime as dt 
    set_json = json.loads(req.get(f'https://api.scryfall.com/sets').text)
    
    return [x['scryfall_uri'].split('/sets/')[1] for x in set_json['data'] if ((dt.strptime(x['released_at'],'%Y-%m-%d') < dt.now()) & (x['set_type'] == 'expansion'))]

# %%


# To Do : 
    # how to visualize deck if no creature types are available (Eldrazi)
    # rarity by card type
    # include lands / card type
    # mana pings for land distribution
    # card type for dual faced cards
    # card type for planeswalker