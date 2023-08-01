import argparse
import os
import sys

c_dir = os.path.dirname(os.path.abspath(__file__))
p_dir = os.path.dirname(c_dir)
sys.path.append(p_dir)

from utils.DB import DB # noqa
from utils.graph import TweetGrah # noqa
from utils.utils import (get_useful_roots, load_tree, # noqa
                         show_or_save, CONFIG) # noqa

db_uri = CONFIG['uri']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera un gráfico de ' +
                                     'con las proporciones de las ' +
                                     'campañas según el número de usos')
    parser.add_argument('-i', '--ignore', action='store_true', default=False,
                        help='Flag to ignore the data in the data folder' +
                        ' and generate it from the database')
    parser.add_argument('-c', '--conversation', type=int, default=None,
                        required=True, help='Conversation id to generate ' +
                        'the sentiment tree')
    args = parser.parse_args()

    # Gets all independent root tweets
    db = DB(db_uri)
    df_root_tweets, df_tweets = get_useful_roots(db, args.ignore)

    tweet = df_root_tweets[
        df_root_tweets['conversation_id'] == args.conversation
    ]
    if tweet.empty:
        sys.exit('Conversation ID is not from a root tweet')

    # Gets the tree of the conversation
    tweet = tweet.iloc[0]
    tree, df_tweets = load_tree(db, tweet, df_tweets)

    # Generates the emotion/sentiment graph
    graph = TweetGrah(tree, rt=False, like=False)
    graph.show_emotion()
    df_stats = graph.get_emotion_stats()
    print(df_stats)

    db.close_connection()
