import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import pickle
import time
import json
import requests as req
import numpy as np
import pandas as pd
import pickle
import pandas as pd
import catboost







def predict(id1, id2):
    r_df = pd.DataFrame()
    d_df = pd.DataFrame()
    r_df[['account_id_1', 'account_id_2', 'account_id_3', 'account_id_4', 'account_id_5',
          'radiant_captain']] = pd.DataFrame(find_team_cap(id1)).T.reset_index(drop=True)
    d_df[['account_id_6', 'account_id_7', 'account_id_8', 'account_id_9', 'account_id_10',
          'dire_captain']] = pd.DataFrame(find_team_cap(id2)).T.reset_index(drop=True)
    res_df = pd.concat([r_df, d_df], axis=1)
    res_df['radiant_team_id'] = id1
    res_df['dire_team_id'] = id2
    res_df['radiant_captain'] = res_df['radiant_captain'].astype(int)
    res_df['dire_captain'] = res_df['dire_captain'].astype(int)

    res_df['r_wins'] = team_wr[id1]['win']
    res_df['d_wins'] = team_wr[id2]['win']

    res_df['r_losses'] = team_wr[id1]['losses']
    res_df['d_losses'] = team_wr[id2]['losses']

    res_df['r_cap_wins'] = capitan_wr[res_df['radiant_captain'].values[0]]['win']
    res_df['d_cap_wins'] = capitan_wr[res_df['dire_captain'].values[0]]['win']

    res_df['r_cap_losses'] = capitan_wr[res_df['radiant_captain'].values[0]]['losses']
    res_df['d_cap_losses'] = capitan_wr[res_df['dire_captain'].values[0]]['losses']

    for i in range(1, 11):
        res_df['account_{}_wins'.format(i)] = account_wr[res_df['account_id_{}'.format(i)].values[0]]['win']
        res_df['account_{}_losses'.format(i)] = account_wr[res_df['account_id_{}'.format(i)].values[0]]['losses']

    res_df['r_rating'] = elo_teams[id1]
    res_df['d_rating'] = elo_teams[id2]

    res_df['r_team_winrate'] = res_df.apply(lambda row: winrate(row['r_wins'], row['r_losses']), axis=1)
    res_df['d_team_winrate'] = res_df.apply(lambda row: winrate(row['d_wins'], row['d_losses']), axis=1)

    res_df['r_capitan_winrate'] = res_df.apply(lambda row: winrate(row['r_cap_wins'], row['r_cap_losses']), axis=1)
    res_df['d_capitan_winrate'] = res_df.apply(lambda row: winrate(row['d_cap_wins'], row['d_cap_losses']), axis=1)
    for i in range(1, 11):
        res_df['account_id_{}_winrate'.format(i)] = res_df.apply(
            lambda row: winrate(row['account_{}_wins'.format(i)], row['account_{}_losses'.format(i)]), axis=1)

    res_df['winrate_team_ratio'] = res_df['r_team_winrate'] / res_df['d_team_winrate']
    res_df['winrate_capitan_ratio'] = res_df['r_capitan_winrate'] / res_df['d_capitan_winrate']
    res_df['sum_r_team_winrate'] = res_df[['account_id_{}_winrate'.format(i) for i in range(1, 6)]].sum(axis=1)
    res_df['sum_d_team_winrate'] = res_df[['account_id_{}_winrate'.format(i) for i in range(6, 11)]].sum(axis=1)
    res_df['sum_winrate_team_ratio'] = res_df['sum_r_team_winrate'] / res_df['sum_d_team_winrate']
    res_df['r_total_cap_games'] = res_df['r_cap_wins'] + res_df['r_cap_losses']
    res_df['d_total_cap_games'] = res_df['d_cap_wins'] + res_df['d_cap_losses']
    res_df['total_r_games'] = res_df[
        ['account_1_wins', 'account_1_losses', 'account_2_wins', 'account_2_losses', 'account_3_wins',
         'account_3_losses',
         'account_4_wins', 'account_4_losses', 'account_5_wins',
         'account_5_losses']].sum(axis=1)
    res_df['total_d_games'] = res_df[['account_6_wins', 'account_6_losses',
                                      'account_7_wins', 'account_7_losses', 'account_8_wins',
                                      'account_8_losses', 'account_9_wins', 'account_9_losses',
                                      'account_10_wins', 'account_10_losses']].sum(axis=1)
    res_df['total_capitan_games_tario'] = res_df['r_total_cap_games'] / res_df['d_total_cap_games']
    res_df['total_players_games_tario'] = res_df['total_r_games'] / res_df['total_d_games']
    res_df['elo_rating_ratio'] = res_df['r_rating'] / res_df['d_rating']
    # res_df = res_df.drop([ 'account_1_wins',
    #        'account_1_losses', 'account_2_wins', 'account_2_losses',
    #        'account_3_wins', 'account_3_losses', 'account_4_wins',
    #        'account_4_losses', 'account_5_wins', 'account_5_losses',
    #        'account_6_wins', 'account_6_losses', 'account_7_wins',
    #        'account_7_losses', 'account_8_wins', 'account_8_losses',
    #        'account_9_wins', 'account_9_losses', 'account_10_wins',
    #        'account_10_losses','account_id_1_winrate', 'account_id_2_winrate',
    #        'account_id_3_winrate', 'account_id_4_winrate', 'account_id_5_winrate',
    #        'account_id_6_winrate', 'account_id_7_winrate', 'account_id_8_winrate',
    #  'account_id_9_winrate', 'account_id_10_winrate','r_cap_wins', 'd_cap_wins', 'r_cap_losses', 'd_cap_losses'],axis = 1)
    res = res_df[['r_wins', 'd_wins', 'r_losses', 'd_losses', 'r_cap_wins', 'd_cap_wins',
                  'r_cap_losses', 'd_cap_losses', 'r_rating', 'd_rating',
                  'account_1_wins', 'account_1_losses', 'account_2_wins',
                  'account_2_losses', 'account_3_wins', 'account_3_losses',
                  'account_4_wins', 'account_4_losses', 'account_5_wins',
                  'account_5_losses', 'account_6_wins', 'account_6_losses',
                  'account_7_wins', 'account_7_losses', 'account_8_wins',
                  'account_8_losses', 'account_9_wins', 'account_9_losses',
                  'account_10_wins', 'account_10_losses', 'r_team_winrate',
                  'd_team_winrate', 'r_capitan_winrate', 'd_capitan_winrate',
                  'account_id_1_winrate', 'account_id_2_winrate', 'account_id_3_winrate',
                  'account_id_4_winrate', 'account_id_5_winrate', 'account_id_6_winrate',
                  'account_id_7_winrate', 'account_id_8_winrate', 'account_id_9_winrate',
                  'account_id_10_winrate', 'winrate_team_ratio', 'winrate_capitan_ratio',
                  'sum_r_team_winrate', 'sum_d_team_winrate', 'sum_winrate_team_ratio',
                  'r_total_cap_games', 'd_total_cap_games', 'total_r_games',
                  'total_d_games', 'total_capitan_games_tario',
                  'total_players_games_tario', 'elo_rating_ratio']]

    return res

def winrate(win,loss):
    if loss+win == 0:
        return 0.47722
    else:
        return win/(loss+win)
def make_row(id1,id2):
    r_df = pd.DataFrame()
    d_df = pd.DataFrame()
    print('228')
    r_df[['account_id_1','account_id_2', 'account_id_3', 'account_id_4', 'account_id_5','radiant_captain']] = pd.DataFrame(find_team_cap(id1)).T.reset_index(drop=True)

    d_df[['account_id_6', 'account_id_7', 'account_id_8', 'account_id_9','account_id_10','dire_captain']] = pd.DataFrame(find_team_cap(id2)).T.reset_index(drop=True)
    res_df = pd.concat([r_df,d_df], axis = 1)
    res_df['radiant_team_id'] = id1
    res_df['dire_team_id'] = id2

    res_df['radiant_captain'] = res_df['radiant_captain'].astype(int)
    res_df['dire_captain'] = res_df['dire_captain'].astype(int)

    res_df['r_wins'] = team_wr[id1]['win']
    res_df['d_wins'] = team_wr[id2]['win']

    res_df['r_losses'] = team_wr[id1]['losses']
    res_df['d_losses'] = team_wr[id2]['losses']

    res_df['r_cap_wins'] = capitan_wr[res_df['radiant_captain'].values[0]]['win']
    res_df['d_cap_wins'] = capitan_wr[res_df['dire_captain'].values[0]]['win']

    res_df['r_cap_losses'] = capitan_wr[res_df['radiant_captain'].values[0]]['losses']
    res_df['d_cap_losses'] = capitan_wr[res_df['dire_captain'].values[0]]['losses']

    for i in range(1,11):
        res_df['account_{}_wins'.format(i)] = account_wr[res_df['account_id_{}'.format(i)].values[0]]['win']
        res_df['account_{}_losses'.format(i)] = account_wr[res_df['account_id_{}'.format(i)].values[0]]['losses']

    res_df['r_rating'] = elo_teams[id1]
    res_df['d_rating'] = elo_teams[id2]

    res_df['r_team_winrate'] = res_df.apply(lambda row:winrate(row['r_wins'],row['r_losses']), axis = 1)
    res_df['d_team_winrate'] = res_df.apply(lambda row:winrate(row['d_wins'],row['d_losses']), axis = 1)

    res_df['r_capitan_winrate'] = res_df.apply(lambda row:winrate(row['r_cap_wins'],row['r_cap_losses']), axis = 1)
    res_df['d_capitan_winrate'] = res_df.apply(lambda row:winrate(row['d_cap_wins'],row['d_cap_losses']), axis = 1)
    for i in range(1,11):
        res_df['account_id_{}_winrate'.format(i)] = res_df.apply(lambda row:winrate(row['account_{}_wins'.format(i)],row['account_{}_losses'.format(i)]), axis = 1)

    res_df['winrate_team_ratio'] = res_df['r_team_winrate']/res_df['d_team_winrate']
    res_df['winrate_capitan_ratio'] = res_df['r_capitan_winrate']/res_df['d_capitan_winrate']
    res_df['sum_r_team_winrate'] = res_df[['account_id_{}_winrate'.format(i)for i in range(1,6)]].sum(axis =1)
    res_df['sum_d_team_winrate'] = res_df[['account_id_{}_winrate'.format(i)for i in range(6,11)]].sum(axis =1)
    res_df['sum_winrate_team_ratio'] = res_df['sum_r_team_winrate']/res_df['sum_d_team_winrate']
    res_df['r_total_cap_games'] = res_df['r_cap_wins'] +res_df['r_cap_losses']
    res_df['d_total_cap_games'] = res_df['d_cap_wins'] +res_df['d_cap_losses']
    res_df['total_r_games'] = res_df[['account_1_wins', 'account_1_losses', 'account_2_wins','account_2_losses', 'account_3_wins', 'account_3_losses',
           'account_4_wins', 'account_4_losses', 'account_5_wins',
           'account_5_losses']].sum(axis =1)
    res_df['total_d_games'] = res_df[['account_6_wins', 'account_6_losses',
           'account_7_wins', 'account_7_losses', 'account_8_wins',
           'account_8_losses', 'account_9_wins', 'account_9_losses',
           'account_10_wins', 'account_10_losses']].sum(axis =1)
    res_df['total_capitan_games_tario']=res_df['r_total_cap_games'] /res_df['d_total_cap_games']
    res_df['total_players_games_tario']=res_df['total_r_games'] /res_df['total_d_games']
    res_df['elo_rating_ratio'] = res_df['r_rating'] / res_df['d_rating']
    # res_df = res_df.drop([ 'account_1_wins',
    #        'account_1_losses', 'account_2_wins', 'account_2_losses',
    #        'account_3_wins', 'account_3_losses', 'account_4_wins',
    #        'account_4_losses', 'account_5_wins', 'account_5_losses',
    #        'account_6_wins', 'account_6_losses', 'account_7_wins',
    #        'account_7_losses', 'account_8_wins', 'account_8_losses',
    #        'account_9_wins', 'account_9_losses', 'account_10_wins',
    #        'account_10_losses','account_id_1_winrate', 'account_id_2_winrate',
    #        'account_id_3_winrate', 'account_id_4_winrate', 'account_id_5_winrate',
    #        'account_id_6_winrate', 'account_id_7_winrate', 'account_id_8_winrate',
    #  'account_id_9_winrate', 'account_id_10_winrate','r_cap_wins', 'd_cap_wins', 'r_cap_losses', 'd_cap_losses'],axis = 1)
    res = res_df[['r_wins', 'd_wins', 'r_losses', 'd_losses', 'r_cap_wins', 'd_cap_wins',
                  'r_cap_losses', 'd_cap_losses', 'r_rating', 'd_rating',
                  'account_1_wins', 'account_1_losses', 'account_2_wins',
                  'account_2_losses', 'account_3_wins', 'account_3_losses',
                  'account_4_wins', 'account_4_losses', 'account_5_wins',
                  'account_5_losses', 'account_6_wins', 'account_6_losses',
                  'account_7_wins', 'account_7_losses', 'account_8_wins',
                  'account_8_losses', 'account_9_wins', 'account_9_losses',
                  'account_10_wins', 'account_10_losses', 'r_team_winrate',
                  'd_team_winrate', 'r_capitan_winrate', 'd_capitan_winrate',
                  'account_id_1_winrate', 'account_id_2_winrate', 'account_id_3_winrate',
                  'account_id_4_winrate', 'account_id_5_winrate', 'account_id_6_winrate',
                  'account_id_7_winrate', 'account_id_8_winrate', 'account_id_9_winrate',
                  'account_id_10_winrate', 'winrate_team_ratio', 'winrate_capitan_ratio',
                  'sum_r_team_winrate', 'sum_d_team_winrate', 'sum_winrate_team_ratio',
                  'r_total_cap_games', 'd_total_cap_games', 'total_r_games',
                  'total_d_games', 'total_capitan_games_tario',
                  'total_players_games_tario', 'elo_rating_ratio']]
    print('END')
    return res


def find_team_cap(id_team):
    for row in pro_matches.iterrows():
        match = row[1]
        if match['game_mode'] == 1:
            continue
        if id_team == match['radiant_team_id']:

            if str(match['radiant_captain']) == 'nan':
                continue
            return match[
                ['account_id_1', 'account_id_2', 'account_id_3', 'account_id_4', 'account_id_5', 'radiant_captain']]
        elif id_team == match['dire_team_id']:

            if str(match['dire_captain']) == 'nan':
                continue
            return match[
                ['account_id_6', 'account_id_7', 'account_id_8', 'account_id_9', 'account_id_10', 'dire_captain']]
    abort(400, description="id2 is None")

st.title('Dota 2 Predictor')
model = pickle.load(open('model.pickle', 'rb'))
pro_matches = pd.read_csv('pro_matches.csv', index_col=0)
team_info = pd.read_csv('team_info.csv', index_col=0)
team_info = team_info.fillna('_')
team_info = team_info.drop(team_info[team_info['name'] == '_'].index)

team_wr = pickle.load(open('team_wr.pickle', 'rb'))
capitan_wr = pickle.load(open('capitan_wr.pickle', 'rb'))
account_wr = pickle.load(open('account_wr.pickle', 'rb'))
elo_teams = pickle.load(open('elo_teams.pickle', 'rb'))

team1 = st.selectbox(
    'Team 1',
    team_info[['name', 'team_id']].values[:1000],index  = 1, format_func=lambda o: o[0])

team2 = st.selectbox(
    'Team 2',
    team_info[['name', 'team_id']].values[:1000],index   = 0, format_func=lambda o: o[0])




if st.button('Predict'):
    with st.spinner('Wait for it...'):
        id1 = team1[1]
        id2 = team2[1]
        x1 = make_row(int(id1), int(id2))
        result = model.predict_proba(x1)

        x2 = make_row(int(id2), int(id1))

        result2 = model.predict_proba(x2)

        resp = {'Team_1': (result[0][1] + result2[0][0]) / 2,
                'Team_2': (result[0][0] + result2[0][1]) / 2,
                'Name_1': team_info.loc[id1]['name'],
                'Name_2': team_info.loc[id2]['name'],
                'id1' : id1,
                'id2' : id2
                }
        st.write(resp)
        st.success('Done!')

