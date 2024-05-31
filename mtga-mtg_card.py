# %%
# deps of sqlalchemy and pandas
# python -m pip install sqlalchemy==1.4.39 pandas

import sqlalchemy as sa
import pandas as pd
import os

# %%

def define_fileconn(mtga_path):
    
    if not mtga_path:
        mtga_path = 'C:\Program Files\Wizards of the Coast\MTGA'
    
    card_db_dir = os.path.join(mtga_path,'MTGA_Data\Downloads\Raw')

    card_db_fn = [x for x in os.listdir(card_db_dir) if 'Raw_CardDatabase' in x][0]

    # the hash at the end of the connection string changes, so this isn't the exact connection string
    eng = sa.create_engine(
        f"sqlite:///{os.path.join(card_db_dir,card_db_fn)}"
    )

    conn = eng.connect()

    return eng, conn

def get_db_df(mtga_path=None,exp_list:list = None):

    conn = define_fileconn(mtga_path = mtga_path)

    if not exp_list:

        query = """
            SELECT DISTINCT ExpansionCode 
            FROM Cards
            """

        exp_list = pd.read_sql(query,conn)

    df = pd.DataFrame()

    for exp in exp_list['ExpansionCode'].to_list():
        query = f"""
            SELECT 
                C.GrpId AS arena_id,
                L.enUS AS name
            FROM Cards C
            LEFT JOIN Localizations L
                ON C.TitleID = L.LocID
            WHERE ExpansionCode LIKE '{exp}'
        """

        df_sub = pd.read_sql(query, conn).drop_duplicates()
        df_sub['Expansion'] = exp
        df = pd.concat([df,df_sub]).reset_index(drop=True)

    return df

# %%
