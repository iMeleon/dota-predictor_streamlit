import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import pickle
import time
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
import json
import requests as req
import numpy as np
import pandas as pd
import pickle
import pandas as pd
import catboost
import trueskill
import itertools
import math
from time import sleep
import torch
import requests
from collections import OrderedDict
def win_probability(team1, team2):
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (4.166666666666667 * 4.166666666666667) + sum_sigma)
    ts = trueskill.global_env()
    return ts.cdf(delta_mu / denom)


def bag_of_heroes(df, N=1, r_val=1, d_val=-1, r_d_val=0, return_as='csr'):
    '''
    Bag of Heroes. Returns a csr matrix (+ list of feature names) or dataframe where each column represents
    a hero (ID) and each row represents a match.

    The value of a cell (i, j) in the returned matrix is:
        cell[i, j] = 0, if the hero or combination of heroes of the j-th column is not present in the i-th match
        cell[i, j] = r_val, if the hero (N = 1) or combination of heroes (N > 1, synergy) of the j-th column is within the Radiant team,
        cell[i, j] = d_val, if the hero (N = 1) or combination of heroes (N > 1, synergy) of the j-th column is within the Dire team,
        cell[i, j] = r_d_val, if the combination of heroes of the j-th column is between the Radiant and Dire teams (N>1, anti-synergy).

    Parameters:
    -----------
        df: dataframe with hero IDs, with columns ['r1_hero_id', ..., 'r5_hero_id', 'd1_hero_id', ..., 'd5_hero_id']
        N: integer 1 <= N <= 10, for N heroes combinations
        return_as: 'csr' for scipy csr sparse matrix, 'df' for pandas dataframe
    '''
    if N < 1 or N > df.shape[1]:
        raise Exception(f'The number N of hero-combinations should be 1 <= N <= {df.shape[1]}')

    # Convert the integer IDs to strings of the form id{x}{x}{x}
    df = df.astype(str).applymap(lambda x: 'id' + '0' * (3 - len(x)) + x)

    # Create a list of all hero IDs present in df
    hero_ids = np.unique(df).tolist()

    # Break df into teams Radiant (r) and Dire (d)
    df_r = df[[col for col in df.columns if col[0] == 'r']]
    df_d = df[[col for col in df.columns if col[0] == 'd']]

    # Create a list of all the hero IDs in df, df_r and df_d respectively
    f = lambda x: ' '.join(['_'.join(c) for c in combinations(sorted(x), N)])

    df_list = df.apply(f, axis=1).tolist()
    df_list.append(' '.join(['_'.join(c) for c in combinations(hero_ids, N)]))

    df_r_list = df_r.apply(f, axis=1).tolist()
    df_r_list.append(' '.join(['_'.join(c) for c in combinations(hero_ids, N)]))

    df_d_list = df_d.apply(f, axis=1).tolist()
    df_d_list.append(' '.join(['_'.join(c) for c in combinations(hero_ids, N)]))

    # Create countvectorizers
    vectorizer = CountVectorizer()
    vectorizer_r = CountVectorizer()
    vectorizer_d = CountVectorizer()

    X = vectorizer.fit_transform(df_list)[:-1]
    X_r = vectorizer_r.fit_transform(df_r_list)[:-1]
    X_d = vectorizer_d.fit_transform(df_d_list)[:-1]
    X_r_d = (X - (X_r + X_d))
    X = (r_val * X_r + d_val * X_d + r_d_val * X_r_d)

    feature_names = vectorizer.get_feature_names()

    if return_as == 'csr':
        return X, feature_names
    elif return_as == 'df':
        return pd.DataFrame(X.toarray(), columns=feature_names, index=df.index)
def get_player_stat(player_id, stat, version=None, hero_id=None):
    if player_id not in player_dict:
        player_dict[player_id] = {
            'wins': 1,
            'losses': 1,
            'rating': env.Rating(),
            'imp': 0,
            'p_wins': 1,
            'p_losses': 1,
            'heroes': {}
        }
    if 'hero' in stat:
        if version not in player_dict[player_id]['heroes']:
            player_dict[player_id]['heroes'][version] = {}
        if hero_id not in player_dict[player_id]['heroes'][version]:
            player_dict[player_id]['heroes'][version][hero_id] = {
                'player_hero_wins': 1,
                'player_hero_losses': 1,
                'player_hero_rating': env.Rating(),
                'player_hero_imp': 0
            }
        if stat == 'player_hero_games':
            return player_dict[player_id]['heroes'][version][hero_id]['player_hero_wins'] + \
                   player_dict[player_id]['heroes'][version][hero_id]['player_hero_losses']
        if stat == 'player_hero_imp':
            if player_dict[player_id]['heroes'][version][hero_id]['player_hero_imp'] == 0:
                return 100
            else:
                return player_dict[player_id]['heroes'][version][hero_id]['player_hero_imp']
        if stat == 'player_hero_rating':
            return player_dict[player_id]['heroes'][version][hero_id]['player_hero_rating'].mu
        return player_dict[player_id]['heroes'][version][hero_id][stat]

    if stat == 'p_games':
        return player_dict[player_id]['p_wins'] + player_dict[player_id]['p_losses']
    if stat == 'games':
        return player_dict[player_id]['wins'] + player_dict[player_id]['losses']
    if stat == 'imp':
        if player_dict[player_id][stat] == 0:
            return 100
        else:
            return player_dict[player_id]['imp']
    if stat == 'rating':
        return player_dict[player_id][stat].mu
    return player_dict[player_id][stat]


def get_hero_stat(hero_id, stat, version):
    if version not in global_heroes:
        global_heroes[version] = {}
    if hero_id not in global_heroes[version]:
        global_heroes[version][hero_id] = {
            'wins': 1,
            'losses': 1,
            'rating': env.Rating(),
        }

    if stat == 'games':
        return global_heroes[version][hero_id]['wins'] + global_heroes[version][hero_id]['losses']
    if stat == 'rating':
        return global_heroes[version][hero_id][stat].mu
    return global_heroes[version][hero_id][stat]


def get_team_stat(team_id, stat):
    if team_id not in team_dict:
        team_dict[team_id] = {
            'wins': 1,
            'losses': 1,
            'rating': env.Rating(),
            'elo_rating': 1000
        }
    if stat == 'rating':
        return team_dict[team_id][stat].mu
    return team_dict[team_id][stat]


def get_captain_stat(captain_id, stat):
    if captain_id not in captain_dict:
        captain_dict[captain_id] = {
            'wins': 1,
            'losses': 1
        }
    return captain_dict[captain_id][stat]


def make_row(id1, id2):
    match = {}
    player_ids = []
    _ = [player_ids.append(team_info.loc[id1]['player{}'.format(i)]) for i in range(1, 6)]
    _= [player_ids.append(team_info.loc[id2]['player{}'.format(i)]) for i in range(1, 6)]
    #     #local_dict_hero = {'hero_{}'.format(idx+1):hero_id for idx, hero_id in enumerate(hero_ids)}
    #     #local_hero_stats = {'global_hero_{}_{}'.format(idx+1,stat):get_hero_stat(hero_id,stat,matches_dict[key]['gameVersionId']) for idx, hero_id in enumerate(hero_ids) for stat in ['wins','losses','rating','games']}
    local_dict_acc = {'account_pro_{}_{}'.format(idx + 1, stat): get_player_stat(player_id, stat) for idx, player_id in
                      enumerate(player_ids) for stat in ['wins', 'losses', 'rating', 'imp', 'games']}
    # local_dict_wins = {'account_{}_{}'.format(idx+1,stat):get_player_stat(player['steamAccountId'],stat,matches_dict[key]['gameVersionId'],player['heroId']) for idx, player in enumerate(matches_dict[key]['players']) for stat in ['player_hero_wins','player_hero_losses','player_hero_rating','player_hero_imp','player_hero_games']}
    local_dict_team_stats = {'{}_team_{}'.format('r' if idx == 0 else 'd', stat): get_team_stat(team_id, stat) for
                             idx, team_id in enumerate([id1, id2]) for stat in
                             ['wins', 'losses', 'rating', 'elo_rating']}
    local_dict_captain_stats = {'{}_captain_{}'.format('r' if idx == 0 else 'd', stat): get_captain_stat(team_id, stat)
                                for idx, team_id in
                                enumerate([team_info.loc[id1]['captain'], team_info.loc[id2]['captain']]) for stat in
                                ['wins', 'losses']}

    t1 = [player_dict[player_id]['rating'] for player_id in player_ids[:5]]
    t2 = [player_dict[player_id]['rating'] for player_id in player_ids[5:]]
    match['pro_players_win_prob'] = win_probability(t1, t2)
    r1 = team_dict[id1]['rating']
    r2 = team_dict[id2]['rating']
    t1 = [r1]
    t2 = [r2]
    match['pro_teams_win_prob'] = win_probability(t1, t2)
    match['elo_pro_teams_win_prob'] = (
                1.0 / (1.0 + pow(10, ((team_dict[id1]['elo_rating'] - team_dict[id2]['elo_rating']) / 400))))
    local_dict_public_stats = {'account_public_{}_{}'.format(idx + 1, stat): get_player_stat(player_id, stat) for
                               idx, player_id in enumerate(player_ids) for stat in ['p_games']}
    match = {**match, **local_dict_acc, **local_dict_team_stats, **local_dict_captain_stats, **local_dict_public_stats}
    df = pd.DataFrame.from_dict(match, orient='index').T
    df['r_team_winrate'] = df['r_team_wins'] / (df['r_team_wins'] + df['r_team_losses'])
    df['d_team_winrate'] = df['d_team_wins'] / (df['d_team_wins'] + df['d_team_losses'])
    df['r_captain_winrate'] = df['r_captain_wins'] / (df['r_captain_wins'] + df['r_captain_losses'])
    df['d_captain_winrate'] = df['d_captain_wins'] / (df['d_captain_wins'] + df['d_captain_losses'])
    for i in range(1, 11):
        # df['account_{}_player_hero_winrate'.format(i)] =df['account_{}_player_hero_wins'.format(i)]/(df['account_{}_player_hero_wins'.format(i)]+ df['account_{}_player_hero_losses'.format(i)])
        df['account_id_{}_pro_winrate'.format(i)] = df['account_pro_{}_wins'.format(i)] / (
                    df['account_pro_{}_wins'.format(i)] + df['account_pro_{}_losses'.format(i)])
        # df['global_hero_{}_winrate'.format(i)] =df['global_hero_{}_wins'.format(i)]/(df['global_hero_{}_wins'.format(i)]+ df['global_hero_{}_losses'.format(i)])

    df['winrate_team_ratio'] = df['r_team_winrate'] / df['d_team_winrate']
    df['winrate_captain_ratio'] = df['r_captain_winrate'] / df['d_captain_winrate']
    # df['sum_r_global_hero_winrate'] = df[['global_hero_{}_winrate'.format(i) for i in range(1, 6)]].sum(axis=1)
    # df['sum_d_global_hero_winrate'] = df[['global_hero_{}_winrate'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['sum_r_account_pro_winrate'] = df[['account_id_{}_pro_winrate'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['sum_d_account_pro_winrate'] = df[['account_id_{}_pro_winrate'.format(i) for i in range(6, 11)]].sum(axis=1)
    # df['sum_r_player_hero_winrate'] = df[['account_{}_player_hero_winrate'.format(i) for i in range(1, 6)]].sum(axis=1)
    # df['sum_d_player_hero_winrate'] = df[['account_{}_player_hero_winrate'.format(i) for i in range(6, 11)]].sum(axis=1)

    df['sum_winrate_account_pro_ratio'] = df['sum_r_account_pro_winrate'] / df['sum_d_account_pro_winrate']
    # df['sum_winrate_global_hero_ratio'] = df['sum_r_global_hero_winrate'] / df['sum_d_global_hero_winrate']
    # df['sum_winrate_player_hero_ratio'] = df['sum_r_player_hero_winrate'] / df['sum_d_player_hero_winrate']

    # df['total_r_player_hero_games'] = df[['account_{}_player_hero_games'.format(i) for i in range(1, 6)]].sum(axis=1)
    # df['total_d_player_hero_games'] = df[['account_{}_player_hero_games'.format(i) for i in range(6, 11)]].sum(axis=1)
    # df['total_r_global_hero_games'] = df[['global_hero_{}_games'.format(i) for i in range(1, 6)]].sum(axis=1)
    # df['total_d_global_hero_games'] = df[['global_hero_{}_games'.format(i) for i in range(6, 11)]].sum(axis=1)
    # df['total_global_hero_games_tario'] = df['total_r_global_hero_games'] / df['total_d_global_hero_games']
    df['r_total_captain_games'] = df['r_captain_wins'] + df['r_captain_losses']
    df['d_total_captain_games'] = df['d_captain_wins'] + df['d_captain_losses']
    df['total_r_pro_games'] = df[['account_pro_{}_games'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['total_d_pro_games'] = df[['account_pro_{}_games'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['total_r_public_games'] = df[['account_public_{}_p_games'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['total_d_public_games'] = df[['account_public_{}_p_games'.format(i) for i in range(6, 11)]].sum(axis=1)

    # df['total_player_hero_tario'] = df['total_r_player_hero_games'] / df['total_d_player_hero_games']
    df['total_captain_games_tario'] = df['r_total_captain_games'] / df['d_total_captain_games']
    df['total_pro_players_games_tario'] = df['total_r_pro_games'] / df['total_d_pro_games']
    df['total_public_players_games_tario'] = df['total_r_public_games'] / df['total_d_public_games']

    df['TS_rating_ratio'] = df['r_team_rating'] / df['d_team_rating']
    df['elo_rating_ratio'] = df['r_team_elo_rating'] / df['d_team_elo_rating']

    df['total_r_TS_pro_rating'] = df[['account_pro_{}_rating'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['total_d_TS_pro_rating'] = df[['account_pro_{}_rating'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['teams_players_pro_rating_TS_ratio'] = df['total_r_TS_pro_rating'] / df['total_d_TS_pro_rating']

    # df['total_r_TS_player_hero_rating'] = df[['account_{}_player_hero_rating'.format(i) for i in range(1, 6)]].sum(axis=1)
    # df['total_d_TS_player_hero_rating'] = df[['account_{}_player_hero_rating'.format(i) for i in range(6, 11)]].sum(axis=1)
    # df['player_hero_rating_TS_ratio'] = df['total_r_TS_player_hero_rating'] / df['total_d_TS_player_hero_rating']

    # df['total_r_TS_global_hero_rating'] = df[['global_hero_{}_rating'.format(i) for i in range(1, 6)]].sum(axis=1)
    # df['total_d_TS_global_hero_rating'] = df[['global_hero_{}_rating'.format(i) for i in range(6, 11)]].sum(axis=1)
    # df['global_hero_rating_TS_ratio'] = df['total_r_TS_global_hero_rating'] / df['total_d_TS_global_hero_rating']

    df['r_imp_pro'] = df[['account_pro_{}_imp'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['d_imp_pro'] = df[['account_pro_{}_imp'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['imp_pro_ratio'] = df['r_imp_pro'] / df['d_imp_pro']

    # df['r_imp_player_hero'] =df[['account_{}_player_hero_imp'.format(i) for i in range(1, 6)]].sum(axis=1)
    # df['d_imp_player_hero'] = df[['account_{}_player_hero_imp'.format(i) for i in range(6, 11)]].sum(axis=1)
    # df['imp_player_hero_ratio']= df['r_imp_player_hero'] /df['d_imp_player_hero']
    return df[['pro_players_win_prob', 'pro_teams_win_prob', 'elo_pro_teams_win_prob',
       'r_team_wins', 'r_team_losses', 'r_team_rating', 'r_team_elo_rating',
       'd_team_wins', 'd_team_losses', 'd_team_rating', 'd_team_elo_rating',
       'r_captain_wins', 'r_captain_losses', 'd_captain_wins',
       'd_captain_losses', 'r_team_winrate', 'd_team_winrate',
       'r_captain_winrate', 'd_captain_winrate', 'winrate_team_ratio',
       'winrate_captain_ratio', 'sum_r_account_pro_winrate',
       'sum_d_account_pro_winrate', 'sum_winrate_account_pro_ratio',
       'r_total_captain_games', 'd_total_captain_games', 'total_r_pro_games',
       'total_d_pro_games', 'total_r_public_games', 'total_d_public_games',
       'total_captain_games_tario', 'total_pro_players_games_tario',
       'total_public_players_games_tario', 'TS_rating_ratio',
       'elo_rating_ratio', 'total_r_TS_pro_rating', 'total_d_TS_pro_rating',
       'teams_players_pro_rating_TS_ratio', 'r_imp_pro', 'd_imp_pro',
       'imp_pro_ratio']]
def make_row_heroes(id1, id2,hero_ids):
    match = {}
    player_ids = []
    _ = [player_ids.append(team_info.loc[id1]['player{}'.format(i)]) for i in range(1, 6)]
    _= [player_ids.append(team_info.loc[id2]['player{}'.format(i)]) for i in range(1, 6)]
    local_dict_hero = {'hero_{}'.format(idx+1):hero_id for idx, hero_id in enumerate(hero_ids)}
    local_hero_stats = {'global_hero_{}_{}'.format(idx+1,stat):get_hero_stat(hero_id,stat,131) for idx, hero_id in enumerate(hero_ids) for stat in ['wins','losses','rating','games']}
    local_dict_acc = {'account_pro_{}_{}'.format(idx + 1, stat): get_player_stat(player_id, stat) for idx, player_id in
                      enumerate(player_ids) for stat in ['wins', 'losses', 'rating', 'imp', 'games']}
    local_dict_wins = {'account_{}_{}'.format(idx+1,stat):get_player_stat(player_id,stat,131,hero_ids[idx]) for idx, player_id in enumerate(player_ids) for stat in ['player_hero_wins','player_hero_losses','player_hero_rating','player_hero_imp','player_hero_games']}
    local_dict_team_stats = {'{}_team_{}'.format('r' if idx == 0 else 'd', stat): get_team_stat(team_id, stat) for
                             idx, team_id in enumerate([id1, id2]) for stat in
                             ['wins', 'losses', 'rating', 'elo_rating']}
    local_dict_captain_stats = {'{}_captain_{}'.format('r' if idx == 0 else 'd', stat): get_captain_stat(team_id, stat)
                                for idx, team_id in
                                enumerate([team_info.loc[id1]['captain'], team_info.loc[id2]['captain']]) for stat in
                                ['wins', 'losses']}

    t1 = [player_dict[player_id]['rating'] for player_id in player_ids[:5]]
    t2 = [player_dict[player_id]['rating'] for player_id in player_ids[5:]]
    match['pro_players_win_prob'] = win_probability(t1, t2)
    r1 = team_dict[id1]['rating']
    r2 = team_dict[id2]['rating']
    t1 = [r1]
    t2 = [r2]
    match['pro_teams_win_prob'] = win_probability(t1, t2)
    match['elo_pro_teams_win_prob'] = (
                1.0 / (1.0 + pow(10, ((team_dict[id1]['elo_rating'] - team_dict[id2]['elo_rating']) / 400))))
    local_dict_public_stats = {'account_public_{}_{}'.format(idx + 1, stat): get_player_stat(player_id, stat) for
                               idx, player_id in enumerate(player_ids) for stat in ['p_games']}
    match = {**match, **local_dict_acc, **local_dict_team_stats, **local_dict_captain_stats, **local_dict_public_stats,**local_dict_hero,**local_hero_stats,**local_dict_wins}
    df = pd.DataFrame.from_dict(match, orient='index').T
    df['r_team_winrate'] = df['r_team_wins'] / (df['r_team_wins'] + df['r_team_losses'])
    df['d_team_winrate'] = df['d_team_wins'] / (df['d_team_wins'] + df['d_team_losses'])
    df['r_captain_winrate'] = df['r_captain_wins'] / (df['r_captain_wins'] + df['r_captain_losses'])
    df['d_captain_winrate'] = df['d_captain_wins'] / (df['d_captain_wins'] + df['d_captain_losses'])
    for i in range(1, 11):
        df['account_{}_player_hero_winrate'.format(i)] =df['account_{}_player_hero_wins'.format(i)]/(df['account_{}_player_hero_wins'.format(i)]+ df['account_{}_player_hero_losses'.format(i)])
        df['account_id_{}_pro_winrate'.format(i)] = df['account_pro_{}_wins'.format(i)] / (
                    df['account_pro_{}_wins'.format(i)] + df['account_pro_{}_losses'.format(i)])
        df['global_hero_{}_winrate'.format(i)] =df['global_hero_{}_wins'.format(i)]/(df['global_hero_{}_wins'.format(i)]+ df['global_hero_{}_losses'.format(i)])

    df['winrate_team_ratio'] = df['r_team_winrate'] / df['d_team_winrate']
    df['winrate_captain_ratio'] = df['r_captain_winrate'] / df['d_captain_winrate']
    df['sum_r_global_hero_winrate'] = df[['global_hero_{}_winrate'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['sum_d_global_hero_winrate'] = df[['global_hero_{}_winrate'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['sum_r_account_pro_winrate'] = df[['account_id_{}_pro_winrate'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['sum_d_account_pro_winrate'] = df[['account_id_{}_pro_winrate'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['sum_r_player_hero_winrate'] = df[['account_{}_player_hero_winrate'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['sum_d_player_hero_winrate'] = df[['account_{}_player_hero_winrate'.format(i) for i in range(6, 11)]].sum(axis=1)

    df['sum_winrate_account_pro_ratio'] = df['sum_r_account_pro_winrate'] / df['sum_d_account_pro_winrate']
    df['sum_winrate_global_hero_ratio'] = df['sum_r_global_hero_winrate'] / df['sum_d_global_hero_winrate']
    df['sum_winrate_player_hero_ratio'] = df['sum_r_player_hero_winrate'] / df['sum_d_player_hero_winrate']

    df['total_r_player_hero_games'] = df[['account_{}_player_hero_games'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['total_d_player_hero_games'] = df[['account_{}_player_hero_games'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['total_r_global_hero_games'] = df[['global_hero_{}_games'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['total_d_global_hero_games'] = df[['global_hero_{}_games'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['total_global_hero_games_tario'] = df['total_r_global_hero_games'] / df['total_d_global_hero_games']
    df['r_total_captain_games'] = df['r_captain_wins'] + df['r_captain_losses']
    df['d_total_captain_games'] = df['d_captain_wins'] + df['d_captain_losses']
    df['total_r_pro_games'] = df[['account_pro_{}_games'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['total_d_pro_games'] = df[['account_pro_{}_games'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['total_r_public_games'] = df[['account_public_{}_p_games'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['total_d_public_games'] = df[['account_public_{}_p_games'.format(i) for i in range(6, 11)]].sum(axis=1)

    df['total_player_hero_tario'] = df['total_r_player_hero_games'] / df['total_d_player_hero_games']
    df['total_captain_games_tario'] = df['r_total_captain_games'] / df['d_total_captain_games']
    df['total_pro_players_games_tario'] = df['total_r_pro_games'] / df['total_d_pro_games']
    df['total_public_players_games_tario'] = df['total_r_public_games'] / df['total_d_public_games']

    df['TS_rating_ratio'] = df['r_team_rating'] / df['d_team_rating']
    df['elo_rating_ratio'] = df['r_team_elo_rating'] / df['d_team_elo_rating']

    df['total_r_TS_pro_rating'] = df[['account_pro_{}_rating'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['total_d_TS_pro_rating'] = df[['account_pro_{}_rating'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['teams_players_pro_rating_TS_ratio'] = df['total_r_TS_pro_rating'] / df['total_d_TS_pro_rating']

    df['total_r_TS_player_hero_rating'] = df[['account_{}_player_hero_rating'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['total_d_TS_player_hero_rating'] = df[['account_{}_player_hero_rating'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['player_hero_rating_TS_ratio'] = df['total_r_TS_player_hero_rating'] / df['total_d_TS_player_hero_rating']

    df['total_r_TS_global_hero_rating'] = df[['global_hero_{}_rating'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['total_d_TS_global_hero_rating'] = df[['global_hero_{}_rating'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['global_hero_rating_TS_ratio'] = df['total_r_TS_global_hero_rating'] / df['total_d_TS_global_hero_rating']

    df['r_imp_pro'] = df[['account_pro_{}_imp'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['d_imp_pro'] = df[['account_pro_{}_imp'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['imp_pro_ratio'] = df['r_imp_pro'] / df['d_imp_pro']

    df['r_imp_player_hero'] =df[['account_{}_player_hero_imp'.format(i) for i in range(1, 6)]].sum(axis=1)
    df['d_imp_player_hero'] = df[['account_{}_player_hero_imp'.format(i) for i in range(6, 11)]].sum(axis=1)
    df['imp_player_hero_ratio']= df['r_imp_player_hero'] /df['d_imp_player_hero']
    col = ['id000', 'id001', 'id002', 'id003', 'id004', 'id005', 'id006', 'id007', 'id008', 'id009', 'id010', 'id011',
           'id012', 'id013', 'id014', 'id015', 'id016', 'id017', 'id018', 'id019', 'id020', 'id021', 'id022', 'id023',
           'id025', 'id026', 'id027', 'id028', 'id029', 'id030', 'id031', 'id032', 'id033', 'id034', 'id035', 'id036',
           'id037', 'id038', 'id039', 'id040', 'id041', 'id042', 'id043', 'id044', 'id045', 'id046', 'id047', 'id048',
           'id049', 'id050', 'id051', 'id052', 'id053', 'id054', 'id055', 'id056', 'id057', 'id058', 'id059', 'id060',
           'id061', 'id062', 'id063', 'id064', 'id065', 'id066', 'id067', 'id068', 'id069', 'id070', 'id071', 'id072',
           'id073', 'id074', 'id075', 'id076', 'id077', 'id078', 'id079', 'id080', 'id081', 'id082', 'id083', 'id084',
           'id085', 'id086', 'id087', 'id088', 'id089', 'id090', 'id091', 'id092', 'id093', 'id094', 'id095', 'id096',
           'id097', 'id098', 'id099', 'id100', 'id101', 'id102', 'id103', 'id104', 'id105', 'id106', 'id107', 'id108',
           'id109', 'id110', 'id111', 'id112', 'id113', 'id114', 'id119', 'id120', 'id121', 'id126', 'id128', 'id129',
           'idnone']
    l_df = pd.DataFrame(0, index=[0], columns=col)
    for hero_id in hero_ids:
        val = 1 if hero_id in [x[0] for x in hero_ids[:5]] else -1
        hero_str_id = str(hero_id)
        if len(hero_str_id) == 1:
            to_add = 'id00'
        elif len(hero_str_id) == 2:
            to_add = 'id0'
        else:
            to_add = 'id'
        l_df.loc[0][to_add + hero_str_id] = val
    print(l_df)
    dum_df = l_df
    res_df = pd.DataFrame()
    res_df_cat = pd.DataFrame()
    model_catboost_dic = pickle.load(open('model_catboost_dic.pickle', 'rb'))
    df['cat_pred'] = model_catboost_dic.predict_proba(dum_df)[:, 1]
    X_p = l_df.values
    X_p = torch.from_numpy(X_p).double()
    model_nn = torch.load('model')
    y_test_pred = model_nn(X_p.float()).detach().cpu().numpy().flatten()
    df['nn_pred'] = y_test_pred[0]
    return df[['pro_players_win_prob', 'pro_teams_win_prob', 'elo_pro_teams_win_prob',
       'r_team_wins', 'r_team_losses', 'r_team_rating', 'r_team_elo_rating',
       'd_team_wins', 'd_team_losses', 'd_team_rating', 'd_team_elo_rating',
       'r_captain_wins', 'r_captain_losses', 'd_captain_wins',
       'd_captain_losses', 'r_team_winrate', 'd_team_winrate',
       'r_captain_winrate', 'd_captain_winrate', 'winrate_team_ratio',
       'winrate_captain_ratio', 'sum_r_account_pro_winrate',
       'sum_d_account_pro_winrate', 'sum_winrate_account_pro_ratio',
       'r_total_captain_games', 'd_total_captain_games', 'total_r_pro_games',
       'total_d_pro_games', 'total_r_public_games', 'total_d_public_games',
       'total_captain_games_tario', 'total_pro_players_games_tario',
       'total_public_players_games_tario', 'TS_rating_ratio',
       'elo_rating_ratio', 'total_r_TS_pro_rating', 'total_d_TS_pro_rating',
       'teams_players_pro_rating_TS_ratio', 'r_imp_pro', 'd_imp_pro',
       'imp_pro_ratio','nn_pred','cat_pred',     'sum_r_global_hero_winrate','sum_d_global_hero_winrate','total_r_global_hero_games','total_d_global_hero_games',
'sum_winrate_global_hero_ratio','total_global_hero_games_tario','total_r_TS_global_hero_rating','total_d_TS_global_hero_rating',
'global_hero_rating_TS_ratio',
 'sum_winrate_player_hero_ratio',
 'total_r_player_hero_games',
 'total_d_player_hero_games',
 'total_player_hero_tario',
 'player_hero_rating_TS_ratio',
 'r_imp_player_hero',
 'd_imp_player_hero',
 'imp_player_hero_ratio']]
heroes = [(1,'Anti-Mage'),(2,'Axe'),(3,'Bane'),(4,'Bloodseeker'),(5,'Crystal Maiden'),(6,'Drow Ranger'),(7,'Earthshaker'),(8,'Juggernaut'),(9,'Mirana'),(10,'Morphling'),(11,'Shadow Fiend'),(12,'Phantom Lancer'),(13,'Puck'),(14,'Pudge'),(15,'Razor'),(16,'Sand King'),(17,'Storm Spirit'),(18,'Sven'),(19,'Tiny'),(20,'Vengeful Spirit'),(21,'Windranger'),(22,'Zeus'),(23,'Kunkka'),(25,'Lina'),(26,'Lion'),(27,'Shadow Shaman'),(28,'Slardar'),(29,'Tidehunter'),(30,'Witch Doctor'),(31,'Lich'),(32,'Riki'),(33,'Enigma'),(34,'Tinker'),(35,'Sniper'),(36,'Necrophos'),(37,'Warlock'),(38,'Beastmaster'),(39,'Queen of Pain'),(40,'Venomancer'),(41,'Faceless Void'),(42,'Wraith King'),(43,'Death Prophet'),(44,'Phantom Assassin'),(45,'Pugna'),(46,'Templar Assassin'),(47,'Viper'),(48,'Luna'),(49,'Dragon Knight'),(50,'Dazzle'),(51,'Clockwerk'),(52,'Leshrac'),(53,"Nature's Prophet"),(54,'Lifestealer'),(55,'Dark Seer'),(56,'Clinkz'),(57,'Omniknight'),(58,'Enchantress'),(59,'Huskar'),(60,'Night Stalker'),(61,'Broodmother'),(62,'Bounty Hunter'),(63,'Weaver'),(64,'Jakiro'),(65,'Batrider'),(66,'Chen'),(67,'Spectre'),(68,'Ancient Apparition'),(69,'Doom'),(70,'Ursa'),(71,'Spirit Breaker'),(72,'Gyrocopter'),(73,'Alchemist'),(74,'Invoker'),(75,'Silencer'),(76,'Outworld Devourer'),(77,'Lycan'),(78,'Brewmaster'),(79,'Shadow Demon'),(80,'Lone Druid'),(81,'Chaos Knight'),(82,'Meepo'),(83,'Treant Protector'),(84,'Ogre Magi'),(85,'Undying'),(86,'Rubick'),(87,'Disruptor'),(88,'Nyx Assassin'),(89,'Naga Siren'),(90,'Keeper of the Light'),(91,'Io'),(92,'Visage'),(93,'Slark'),(94,'Medusa'),(95,'Troll Warlord'),(96,'Centaur Warrunner'),(97,'Magnus'),(98,'Timbersaw'),(99,'Bristleback'),(100,'Tusk'),(101,'Skywrath Mage'),(102,'Abaddon'),(103,'Elder Titan'),(104,'Legion Commander'),(105,'Techies'),(106,'Ember Spirit'),(107,'Earth Spirit'),(108,'Underlord'),(109,'Terrorblade'),(110,'Phoenix'),(111,'Oracle'),(112,'Winter Wyvern'),(113,'Arc Warden'),(114,'Monkey King'),(119,'Dark Willow'),(120,'Pangolier'),(121,'Grimstroke'),(126,'Void Spirit'),(128,'Snapfire'),(129,'Mars')]
col  = ['id000','id001','id002','id003','id004','id005','id006','id007','id008','id009','id010','id011','id012','id013','id014','id015','id016','id017','id018','id019','id020','id021','id022','id023','id025','id026','id027','id028','id029','id030','id031','id032','id033','id034','id035','id036','id037','id038','id039','id040','id041','id042','id043','id044','id045','id046','id047','id048','id049','id050','id051','id052','id053','id054','id055','id056','id057','id058','id059','id060','id061','id062','id063','id064','id065','id066','id067','id068','id069','id070','id071','id072','id073','id074','id075','id076','id077','id078','id079','id080','id081','id082','id083','id084','id085','id086','id087','id088','id089','id090','id091','id092','id093','id094','id095','id096','id097','id098','id099','id100','id101','id102','id103','id104','id105','id106','id107','id108','id109','id110','id111','id112','id113','id114','id119','id120','id121','id126','id128','id129','idnone']

@st.cache(suppress_st_warning=True)
def get_players_info(players):
    res = []
    dict_players_id = pickle.load(open('dict_players_id.pickle', 'rb'))
    for id in players:
        if id not in dict_players_id:
            url = 'https://api.opendota.com/api/players/{}'.format(id)
            resp = requests.get(url, timeout=20)
            load_resp = json.loads(resp.text)
            res.append((load_resp['profile']['account_id'], load_resp['profile']['name'] if 'name' in load_resp['profile'] else 'Name not found'))
            dict_players_id[id] = {}
            dict_players_id[id]['account_id'] = load_resp['profile']['account_id']
            dict_players_id[id]['name'] = load_resp['profile']['name'] if 'name' in load_resp['profile'] else 'Name not found'
        else:
            res.append(( dict_players_id[id]['account_id'],dict_players_id[id]['name'] ))
    with open('dict_players_id.pickle', 'wb') as f2:
        pickle.dump(dict_players_id, f2)
    return res
def test1():
    p_type = st.radio(
    "Predict type", ('Without picks', 'With picks'))
    team1 = st.selectbox(
        'Team 1',
        team_info[['name', 'id']].values[:1000], index=1, format_func=lambda o: o[0])

    team2 = st.selectbox(
        'Team 2',
        team_info[['name', 'id']].values[:1000], index=0, format_func=lambda o: o[0])

    # Add a selectbox to the sidebar:
    if p_type == 'Without picks':
        if st.button('Predict'):
            id1 = team1[1]
            id2 = team2[1]
            model = pickle.load(open('model.pickle', 'rb'))
            x1 = make_row(int(id1), int(id2))
            result = model.predict_proba(x1)

            x2 = make_row(int(id2), int(id1))

            result2 = model.predict_proba(x2)

            resp = {'Team_1': (result[0][1] + result2[0][0]) / 2,
                        'Team_2': (result[0][0] + result2[0][1]) / 2,
                        'Name_1': team_info.loc[id1]['name'],
                        'Name_2': team_info.loc[id2]['name'],
                        'id1': id1,
                        'id2': id2
                        }
            st.write(resp)
            st.success('Done!')
    else:
        id1 = team1[1]
        id2 = team2[1]
        r_players = team_info.loc[team1[1]][['player1', 'player2', 'player3', 'player4', 'player5']].values
        r = get_players_info(r_players)
        st.markdown("""
        Player 1 id: **{}** name: **{}**
        
        Player 2 id: **{}** name: **{}**
        
        Player 3 id: **{}** name: **{}**
        
        Player 4 id: **{}** name: **{}**
        
        Player 5 id: **{}** name: **{}**
        """.format(r[0][0],r[0][1],r[1][0],r[1][1],r[2][0],r[2][1],r[3][0],r[3][1],r[4][0],r[4][1]))
        r_heroes = st.multiselect('Pick radiant heroes in right order',heroes, format_func=lambda o: o[1])

        d_players = team_info.loc[team2[1]][['player1', 'player2', 'player3', 'player4', 'player5']].values
        d = get_players_info(d_players)
        st.markdown("""
        Player 1 id: **{}** name: **{}**

        Player 2 id: **{}** name: **{}**

        Player 3 id: **{}** name: **{}**

        Player 4 id: **{}** name: **{}**

        Player 5 id: **{}** name: **{}**
        """.format(d[0][0], d[0][1], d[1][0], d[1][1], d[2][0], d[2][1], d[3][0], d[3][1], d[4][0], d[4][1]))
        d_heroes = st.multiselect('Pick dire heroes in right order', heroes, format_func=lambda o: o[1])

        if st.button('Predict'):
            if len([*r_heroes,*d_heroes]) != 10:
                st.warning("Pick heroes 10 heroes pls")
            else:
                x1 = make_row_heroes(int(id1), int(id2),[*r_heroes,*d_heroes])
                last_model = pickle.load(open('last_model.pickle', 'rb'))
                result = last_model.predict_proba(x1)

                resp = {'Team_1': result[0][1],
                        'Team_2': result[0][0],
                        'Name_1': team_info.loc[id1]['name'],
                        'Name_2': team_info.loc[id2]['name'],
                        'id1': id1,
                        'id2': id2
                        }
                st.write(resp)
                st.success('Done!')

def test2():
    heroes_team_1 = st.multiselect('Radiant heroes',
        heroes, format_func=lambda o: o[1])
    heroes_team_2 = st.multiselect('Dire heroes',
        heroes, format_func=lambda o: o[1])
    if st.button('Predict'):
        arr = [*[x[0] for x in heroes_team_1], *[x[0] for x in heroes_team_2]]
        l_df = pd.DataFrame(0, index=[0], columns=col)
        for  hero_id in arr:
            val = 1 if hero_id in [x[0] for x in heroes_team_1] else -1
            hero_str_id = str(hero_id)
            if len(hero_str_id) == 1:
                to_add = 'id00'
            elif len(hero_str_id) == 2:
                to_add = 'id0'
            else:
                to_add = 'id'
            l_df.loc[0][to_add + hero_str_id] = val
        model_heroes_catboost = pickle.load(open('model_heroes_catboost.pickle', 'rb'))
        pred_cat_heroes = model_heroes_catboost.predict_proba(l_df)
        st.write('Radiant win prob(Catboost): ' + str(pred_cat_heroes[0][1]))
        X_p = l_df.values
        X_p = torch.from_numpy(X_p).double()
        model_nn = torch.load('model')
        y_test_pred = model_nn(X_p.float()).detach().cpu().numpy().flatten()
        st.write('Radiant win prob(PyTorch): '+str(y_test_pred[0]))

        st.success('Done!')
st.title('Dota 2 Predictor')
env = trueskill.TrueSkill(draw_probability=0)
env.make_as_global()


main_dict = pickle.load(open('main_dict.pickle', 'rb'))
player_dict = main_dict['player_dict']
team_dict = main_dict['team_dict']
captain_dict = main_dict['captain_dict']
global_heroes = main_dict['global_heroes']
max_id = main_dict['max_id']
team_info = pd.DataFrame(team_dict).T.sort_values(by=['elo_rating'], ascending= False)


DEMOS = OrderedDict(
    [
        ("Teams", (test1, None)),
        (
            "Heroes",
            (
                test2,
                """
This app shows how you can use Streamlit to build cool animations.
It displays an animated fractal based on the the Julia Set. Use the slider
to tune different parameters.
""",
            ),
        )
    ]
)
def run():
    demo_name = st.sidebar.selectbox("Choose a demo", list(DEMOS.keys()), 0)
    demo = DEMOS[demo_name][0]
    for i in range(10):
        st.empty()

    demo()




if __name__ == "__main__":
    run()


