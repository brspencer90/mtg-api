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

def dsk_keywords(keywords,type_line,clean_text,double_face=False):
    if double_face:

    else:
        # Delerium - RGWB - mill, graveyard, surveil, manifest dread
            # RG - Delerium stompy
            # WB - Reanimator
            # GB - Manifest dread

        # Eerie - UBWR - enchantment, rooms
            # BU - control
            # WU - tempo
            # RU - rooms

        # Survival - GW - 

        # Sacrifice - RB 

        # Aggro - RW