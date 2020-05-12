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
import trueskill
import itertools
import math
from time import sleep
# import torch
from collections import OrderedDict
def win_probability(team1, team2):
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    denom = math.sqrt(size * (4.166666666666667 * 4.166666666666667) + sum_sigma)
    ts = trueskill.global_env()
    return ts.cdf(delta_mu / denom)


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
heroes = [(1,'Anti-Mage'),(2,'Axe'),(3,'Bane'),(4,'Bloodseeker'),(5,'Crystal Maiden'),(6,'Drow Ranger'),(7,'Earthshaker'),(8,'Juggernaut'),(9,'Mirana'),(10,'Morphling'),(11,'Shadow Fiend'),(12,'Phantom Lancer'),(13,'Puck'),(14,'Pudge'),(15,'Razor'),(16,'Sand King'),(17,'Storm Spirit'),(18,'Sven'),(19,'Tiny'),(20,'Vengeful Spirit'),(21,'Windranger'),(22,'Zeus'),(23,'Kunkka'),(25,'Lina'),(26,'Lion'),(27,'Shadow Shaman'),(28,'Slardar'),(29,'Tidehunter'),(30,'Witch Doctor'),(31,'Lich'),(32,'Riki'),(33,'Enigma'),(34,'Tinker'),(35,'Sniper'),(36,'Necrophos'),(37,'Warlock'),(38,'Beastmaster'),(39,'Queen of Pain'),(40,'Venomancer'),(41,'Faceless Void'),(42,'Wraith King'),(43,'Death Prophet'),(44,'Phantom Assassin'),(45,'Pugna'),(46,'Templar Assassin'),(47,'Viper'),(48,'Luna'),(49,'Dragon Knight'),(50,'Dazzle'),(51,'Clockwerk'),(52,'Leshrac'),(53,"Nature's Prophet"),(54,'Lifestealer'),(55,'Dark Seer'),(56,'Clinkz'),(57,'Omniknight'),(58,'Enchantress'),(59,'Huskar'),(60,'Night Stalker'),(61,'Broodmother'),(62,'Bounty Hunter'),(63,'Weaver'),(64,'Jakiro'),(65,'Batrider'),(66,'Chen'),(67,'Spectre'),(68,'Ancient Apparition'),(69,'Doom'),(70,'Ursa'),(71,'Spirit Breaker'),(72,'Gyrocopter'),(73,'Alchemist'),(74,'Invoker'),(75,'Silencer'),(76,'Outworld Devourer'),(77,'Lycan'),(78,'Brewmaster'),(79,'Shadow Demon'),(80,'Lone Druid'),(81,'Chaos Knight'),(82,'Meepo'),(83,'Treant Protector'),(84,'Ogre Magi'),(85,'Undying'),(86,'Rubick'),(87,'Disruptor'),(88,'Nyx Assassin'),(89,'Naga Siren'),(90,'Keeper of the Light'),(91,'Io'),(92,'Visage'),(93,'Slark'),(94,'Medusa'),(95,'Troll Warlord'),(96,'Centaur Warrunner'),(97,'Magnus'),(98,'Timbersaw'),(99,'Bristleback'),(100,'Tusk'),(101,'Skywrath Mage'),(102,'Abaddon'),(103,'Elder Titan'),(104,'Legion Commander'),(105,'Techies'),(106,'Ember Spirit'),(107,'Earth Spirit'),(108,'Underlord'),(109,'Terrorblade'),(110,'Phoenix'),(111,'Oracle'),(112,'Winter Wyvern'),(113,'Arc Warden'),(114,'Monkey King'),(119,'Dark Willow'),(120,'Pangolier'),(121,'Grimstroke'),(126,'Void Spirit'),(128,'Snapfire'),(129,'Mars')]
col  = ['id000','id001','id002','id003','id004','id005','id006','id007','id008','id009','id010','id011','id012','id013','id014','id015','id016','id017','id018','id019','id020','id021','id022','id023','id025','id026','id027','id028','id029','id030','id031','id032','id033','id034','id035','id036','id037','id038','id039','id040','id041','id042','id043','id044','id045','id046','id047','id048','id049','id050','id051','id052','id053','id054','id055','id056','id057','id058','id059','id060','id061','id062','id063','id064','id065','id066','id067','id068','id069','id070','id071','id072','id073','id074','id075','id076','id077','id078','id079','id080','id081','id082','id083','id084','id085','id086','id087','id088','id089','id090','id091','id092','id093','id094','id095','id096','id097','id098','id099','id100','id101','id102','id103','id104','id105','id106','id107','id108','id109','id110','id111','id112','id113','id114','id119','id120','id121','id126','id128','id129','idnone']

def test1():
    team1 = st.selectbox(
        'Team 1',
        team_info[['name', 'id']].values[:1000], index=1, format_func=lambda o: o[0])

    team2 = st.selectbox(
        'Team 2',
        team_info[['name', 'id']].values[:1000], index=0, format_func=lambda o: o[0])

    # Add a selectbox to the sidebar:

    if st.button('Predict'):
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
        for idx, hero_id in enumerate(arr):
            val = 1 if idx < 5 else -1
            hero_str_id = str(hero_id)
            if len(hero_str_id) == 1:
                to_add = 'id00'
            elif len(hero_str_id) == 2:
                to_add = 'id0'
            else:
                to_add = 'id'
            l_df.loc[0][to_add + hero_str_id] = val
        X_p = l_df.values
        st.write(X_p)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # X_p = torch.from_numpy(X_p).double()
        # X_p = X_p.to(device)
        # model = torch.load('model')
        # y_test_pred = model(X_p.float()).detach().cpu().numpy().flatten()
        # st.write('Radiant win prob: '+str(y_test_pred[0]))

        st.success('Done!')
st.title('Dota 2 Predictor')
model = pickle.load(open('model.pickle', 'rb'))
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


