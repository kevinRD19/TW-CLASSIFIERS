import argparse
import os
import sys

c_dir = os.path.dirname(os.path.abspath(__file__))
p_dir = os.path.dirname(c_dir)
sys.path.append(p_dir)

from utils.graph import TweetGrah # noqa
from utils.DB import DB # noqa
from utils.utils import (load_tree, get_useful_roots, # noqa
                         CONFIG, show_or_save) # noqa


db_uri = CONFIG['uri']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera un gráfico de ' +
                                     'con las proporciones de las ' +
                                     'campañas según el número de usos')
    parser.add_argument('-i', '--ignore', action='store_true', default=False,
                        help='Flag to ignore the data in the data folder' +
                        ' and generate a new one from the database')
    group = parser.add_argument_group('Execution mode. Default over all ' +
                                      'root tweets')
    mode = group.add_mutually_exclusive_group()
    mode.add_argument('-c', '--conversation', type=int,
                      help='Conversation id to generate the conversation tree')
    mode.add_argument('-n', '--numconv', type=int,
                      help='Number of conversations to generate the trees')
    mode.add_argument('-v', '--verbose', action='store_true', default=False,
                      help='Indicates if the program should show the ' +
                      'information in the nodes of the graph (id, text, ' +
                      'author)')
    args = parser.parse_args()

    db = DB(db_uri)
    df_root_tweets, df_tweets = get_useful_roots(db, args.ignore, True)

    os.makedirs('data/tree5', exist_ok=True)

    if args.conversation:
        roots = df_root_tweets[
            df_root_tweets['conversation_id'] == args.conversation
            ]
        if roots.empty:
            sys.exit('Conversation ID is not from a root tweet')
        roots = roots[:1]
    elif args.numconv:
        df_root_tweets = df_root_tweets[:args.numconv]
    else:
        roots = df_root_tweets

    os.makedirs('data/tree', exist_ok=True)
    for _, root in roots.iterrows():
        tree, df_tweets = load_tree(db, root, df_tweets)
        graph = TweetGrah(tree)
        graph.show(args.verbose)

    db.close_connection()
