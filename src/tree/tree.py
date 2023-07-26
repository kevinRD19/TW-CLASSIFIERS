import argparse
import os
import sys


c_dir = os.path.dirname(os.path.abspath(__file__))
p_dir = os.path.dirname(c_dir)
sys.path.append(p_dir)

from utils.DB import DB # noqa
from utils.utils import (get_useful_roots, load_tree, # noqa
                         CONFIG, show_or_save) # noqa

db_uri = CONFIG['uri']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera un gráfico de ' +
                                     'con las proporciones de las ' +
                                     'campañas según el número de usos')
    parser.add_argument('-i', '--ignore', action='store_true', default=False,
                        help='Flag to ignore the data in the data folder' +
                        ' and generate a new one from the database')
    parser.add_argument('-c', '--conversation', type=int, nargs='+',
                        help='Conversation ids to generate the tree JSON file')
    args = parser.parse_args()

    db = DB(db_uri)
    df_root_tweets, df_tweets = get_useful_roots(db, args.ignore, True)

    os.makedirs('data/tree5', exist_ok=True)
    conversations = []
    files = os.listdir('data/tree/') + os.listdir('data/tree2/') + \
        os.listdir('data/tree3/') + os.listdir('data/tree4/') + \
        os.listdir('data/tree5/')

    for _file in files:
        conversation_id = int(_file.split('.')[0])
        conversations.append(conversation_id)

    df_root_tweets = df_root_tweets.loc[
        ~df_root_tweets['conversation_id'].isin(conversations)
    ].copy().reset_index(drop=True)

    roots = df_root_tweets.loc[
        df_root_tweets['conversation_id'].isin(args.conversation)
    ] if args.conversation else df_root_tweets

    for _, root in roots.iterrows():
        load_tree(db, root, df_tweets)
    db.close_connection()
