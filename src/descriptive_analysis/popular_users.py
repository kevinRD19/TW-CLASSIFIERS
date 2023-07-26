import argparse
from functools import reduce
import os
import sys
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


c_dir = os.path.dirname(os.path.abspath(__file__))
p_dir = os.path.dirname(c_dir)
sys.path.append(p_dir)

from utils.DB import DB # noqa
from utils.utils import (Radar, get_useful_roots, load_tree, # noqa
                         load_root_tweets, get_dates, # noqa
                         CONFIG, show_or_save) # noqa
from utils.graph import TweetGrah # noqa

db_uri = CONFIG['uri']


def get_non_calculated(root_tweets: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    if 'graph_statics.csv' not in os.listdir('data'):
        data = {
            'tweet_id': [], 'conversation_id': [], 'author_id': [],
            'start': [], 'end': [], 'max_depht': [], 'num_tweets': []
        }
        return root_tweets, data
    else:
        df_graph_statics = pd.read_csv('data/graph_statics.csv')
        return root_tweets.loc[
            ~root_tweets['conversation_id'].
            isin(df_graph_statics['conversation_id'])
        ].copy().reset_index(drop=True), df_graph_statics.to_dict('list')


def add_data(data: dict, root: pd.Series, graph: TweetGrah) -> None:
    statics = graph.get_statics()
    [
        data[field].append(root[field])
        for field in ['tweet_id', 'conversation_id', 'author_id']
    ]
    [
        data[key].append(value) for key, value in statics.items()
    ]


def get_user_statics(db, ignore):
    if 'user_statics.csv' in os.listdir('data') and not ignore:
        return pd.read_csv('data/user_statics.csv')

    df_root_tweets, df_tweets = get_useful_roots(db, args.ignore, True)

    roots = df_root_tweets.loc[
        df_root_tweets['conversation_id'].isin(args.conversation)
    ] if args.conversation else df_root_tweets

    roots, data = get_non_calculated(roots)

    for _, root in roots.iterrows():
        tree, df_tweets = load_tree(db, root, df_tweets)
        graph = TweetGrah(tree, rt=False, like=False)
        add_data(data, root, graph)

    df_graph_statics = pd.DataFrame(data)
    df_graph_statics.to_csv('data/graph_statics.csv', index=False)

    df_graph_statics.drop(columns=['tweet_id', 'author_id'], inplace=True)
    df_root_tweets = load_root_tweets(db, ignore=args.ignore)
    df_root_tweets = df_root_tweets.loc[
        df_root_tweets['lang'] == 'es'
        ][['id', 'conversation_id']]

    df_tweet_date = get_dates(db, ignore=args.ignore)
    df_root_tweets = df_root_tweets.merge(df_tweet_date, how='left',
                                          on='id')
    df_root_tweets = df_root_tweets.merge(df_graph_statics, how='outer',
                                          on='conversation_id')
    df_root_tweets.loc[
        df_root_tweets.isna().any(axis=1), ['max_depht', 'num_tweets']
        ] = 1
    df_root_tweets['start'] = np.where(df_root_tweets['start'].isna(),
                                       df_root_tweets['created_at'],
                                       df_root_tweets['start'])
    df_root_tweets['end'] = np.where(df_root_tweets['end'].isna(),
                                     df_root_tweets['created_at'],
                                     df_root_tweets['end'])
    df_root_tweets['start'] = pd.to_datetime(df_root_tweets['start'])
    df_root_tweets['end'] = pd.to_datetime(df_root_tweets['end'])
    df_root_tweets['duration'] = df_root_tweets['end'] - \
        df_root_tweets['start']

    conversations = df_root_tweets.groupby(by=['author_id']).size()\
        .reset_index(name='num_conversations')
    means = df_root_tweets[['author_id', 'max_depht', 'num_tweets']]. \
        groupby(by=['author_id']).mean().reset_index()
    durations = df_root_tweets[['author_id', 'duration']].groupby(
        by=['author_id']).mean().reset_index()
    df_root_tweets = conversations.merge(means, how='right', on='author_id')
    df_root_tweets = df_root_tweets.merge(durations, how='right',
                                          on='author_id')

    df_root_tweets.to_csv('data/user_statics.csv', index=False)

    return df_root_tweets


def radar_chart(df_tweets, sortby=['num_tweets']):
    df_tweets.sort_values(by=sortby, ascending=[False]*len(sortby),
                          inplace=True)

    tweets = df_tweets[:8].copy().reset_index(drop=True)
    columns = np.delete(tweets.columns, [0])
    fig = plt.figure(figsize=(8, 8))

    titles = ['Num Convertations', 'Max Depht', 'Num Tweets', 'Duration']
    labels = []
    for c in columns:
        _max = tweets[c].max()
        _max = _max if _max else df_tweets[c].max()
        labels.append(np.around(np.arange(0, 6/5*_max, (6/5*_max)/6), 2)[-5:])
        tweets[c] = (tweets[c]*5/_max).round(2)

    radar = Radar(fig, titles, labels)
    colors = ['b', 'g', 'r', 'darkorange', 'y', 'k', 'grey', 'pink',
              'm', 'c']
    for _, row in tweets.iterrows():
        radar.plot(row[columns],  '-', lw=3, alpha=1,
                   label=row['author_id'].astype(int), color=colors[_])
    radar.ax.legend(title='User ID', fontsize=26, shadow=True, draggable=True,
                    bbox_to_anchor=(1.45, 0.6),
                    title_fontproperties={'weight': 'bold',
                                          'size': 30})
    fig.subplots_adjust(top=1.1, bottom=-0.7)
    fig.set_size_inches(32, 18)
    os.makedirs('images/users/popular', exist_ok=True)
    last_name = reduce(lambda x, y: ''.join(x.split('_')).capitalize() +
                       ''.join(y.split('_')).capitalize(), sortby) \
        if len(sortby) > 1 else \
        ''.join(sortby[0].split('_')).capitalize()
    plt.savefig(f'images/users/popular/{last_name}.png', dpi=350)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera un gráfico de ' +
                                     'con las proporciones de las ' +
                                     'campañas según el número de usos')
    parser.add_argument('-i', '--ignore', action='store_true', default=False,
                        help='Flag to ignore the data in the data folder' +
                        ' and generate a new one from the database')
    args = parser.parse_args()

    db = DB(db_uri)

    df_root_tweets = get_user_statics(db, args.ignore)
    print(df_root_tweets)
    df_root_tweets['duration'] = df_root_tweets['duration']. \
        astype('timedelta64[D]').dt.days

    df_root_tweets = df_root_tweets.loc[df_root_tweets['num_tweets'] >= 5]
    df_root_tweets = df_root_tweets.loc[
        df_root_tweets['num_conversations'] >= 10]

    radar_chart(df_root_tweets, ['duration', 'num_tweets'])

    db.close_connection()
