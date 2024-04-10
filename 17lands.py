# %%
import requests as r
import pandas as pd

from datetime import datetime as dt

from constants import Constants as c

import logging

logging.basicConfig(level=logging.INFO)
# %%
def update_files(EXPANSION:str,FORMAT:str='PremierDraft'):

    assert EXPANSION in c.expansion_list, 'invalid expansion value, please check input and try again'
    assert FORMAT in ['PremierDraft','TradDraft'], "invalid format, must be in ['PremierDraft','TradDraft']"

    # Choose the MTG set. This is the 3 letter code for a set

    FORMAT_PREMIER_DRAFT = "PremierDraft"
    FORMAT_TRADITIONAL_DRAFT = "TradDraft"

    # Choose the format to analyze
    FORMAT = FORMAT_PREMIER_DRAFT
    GAME_DATA_FILE = f"game_data_public.{EXPANSION}.{FORMAT}.csv.gz"
    GAME_DATA_REMOTE_URL = f"https://17lands-public.s3.amazonaws.com/analysis_data/game_data/{GAME_DATA_FILE}"
    DRAFT_DATA_FILE = f"draft_data_public.{EXPANSION}.{FORMAT}.csv.gz"
    DRAFT_DATA_REMOTE_URL = f"https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/{DRAFT_DATA_FILE}"

    local_game_fn = f'{EXPANSION}_{FORMAT}_GAME.csv.gz'
    local_draft_fn = f'{EXPANSION}_{FORMAT}_DRAFT.csv.gz'
    local_card_fn = f'card_data.csv'

    time_i = dt.now()
    logging.info(f'downloading game data file for {EXPANSION}')
    open(local_game_fn,'wb').write(r.get(GAME_DATA_REMOTE_URL,stream=True).content)

    logging.info(f'finished downloading game data file - {(dt.now() - time_i).seconds}s, downloading draft data')
    time_i = dt.now()

    open(local_draft_fn,'wb').write(r.get(DRAFT_DATA_REMOTE_URL,stream=True).content)

    logging.info(f'finished downloading draft data file - {(dt.now() - time_i).seconds}s, downloading card data')
    time_i = dt.now()

    open(local_card_fn,'wb').write(r.get('https://17lands-public.s3.amazonaws.com/analysis_data/cards/cards.csv',stream=True).content)

    logging.info(f'finished downloading card data file - {(dt.now() - time_i).seconds}s')

# %%

df = pd.read_csv('MKM_PremierDraft_GAME.csv.gz',chunksize=1000)
df = pd.concat(list(df)[:1],ignore_index=True)


def get_game_drawn_cols():
    """
    Returns the columns in the game data file that includes metadata and which cards were in the deck. Filters out other columns to reduce size of the dataset
    """
    df = next(pd.read_csv('MKM_PremierDraft_GAME.csv.gz', chunksize=100))
    col_names = list(df)
    gd_card_cols = [x for x in col_names if x.startswith("drawn_")]

    gd_base_cols = ['draft_id', 'main_colors', 'splash_colors', 'num_turns', 'won']
    gd_all_cols = gd_base_cols + gd_card_cols
    return gd_all_cols

17 lands analytics
- filter by ever in hand
*** gih wr