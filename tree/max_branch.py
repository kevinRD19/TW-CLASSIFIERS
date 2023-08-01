import argparse
import os
import sys
import pandas as pd
from pandas import DataFrame
import networkx as nx

c_dir = os.path.dirname(os.path.abspath(__file__))
p_dir = os.path.dirname(c_dir)
sys.path.append(p_dir)

from utils.graph import TweetGrah # noqa
from utils.DB import DB # noqa
from utils.utils import (load_tree, get_useful_roots, # noqa
                         CONFIG, show_or_save) # noqa


db_uri = CONFIG['uri']


def tweets() -> DataFrame:
    """
    Get the tweets to generate the tree

    Returns
    -------
        : _description_
    """
    os.makedirs('data', exist_ok=True)
    if 'tweets_conexion.csv' not in os.listdir('data') or args.ignore:
        df_tweets = db.get_tweets_to_conexion()
        df_count = df_tweets.groupby(by=['conversation_id']).size()\
                            .reset_index(name='num_tweets')
        df_tweets = df_tweets.merge(df_count, how='left', on='conversation_id')
        df_tweets.to_csv('data/tweets_conexion.csv', index=False)
    else:
        df_tweets = pd.read_csv('data/tweets_conexion.csv')

    return df_tweets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera un gráfico de ' +
                                     'con las proporciones de las ' +
                                     'campañas según el número de usos')
    parser.add_argument('-i', '--ignore', action='store_true', default=False,
                        help='Flag to ignore the data in the data folder' +
                        ' and generate it from the database')
    parser.add_argument('-c', '--conversation', type=int,
                        help='Conversation id to generate the maximum ' +
                        'length branch conversation tree')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Indicates if the program should show the ' +
                        'information in the nodes of the graph (id, text, ' +
                        'author)')
    args = parser.parse_args()

    # Gets all independent root tweets
    db = DB(db_uri)
    df_root_tweets, df_tweets = get_useful_roots(db, args.ignore)

    if args.conversation:
        tweet = df_root_tweets[
            df_root_tweets['conversation_id'] == args.conversation
            ]
        if tweet.empty:
            sys.exit('Conversation ID is not from a root tweet')
        roots = df_root_tweets[:1]
    else:
        tweet = df_root_tweets.sample(1)

    tweet = tweet.iloc[0]
    tree, df_tweets = load_tree(db, tweet, df_tweets)

    graph = TweetGrah(tree, rt=False, like=False)

    # Obtains the maximum length branch
    path = graph.get_max_branch_nodes()
    subgraph = nx.induced_subgraph(graph, path)
    subgraph.show(args.verbose)

    # Shows some statistics of the deepest branch
    df_statics = graph.get_statics()

    db.close_connection()
