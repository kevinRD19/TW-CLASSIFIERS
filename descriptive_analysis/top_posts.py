import argparse
from datetime import datetime
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

c_dir = os.path.dirname(os.path.abspath(__file__))
p_dir = os.path.dirname(c_dir)
sys.path.append(p_dir)

from utils.utils import CONFIG, show_or_save # noqa
from utils.DB import DB # noqa

db_uri = CONFIG['uri']


def bar_comparation(metric: str):
    """
    Create a bar plot with the top 10 posts with more interactions in
    the defined metric.

    Arguments
    ----------
        - metric (`str`): tweet metric to compare.
    """
    df_top = df_tw_interactions.sort_values(by=[metric],
                                            ascending=False).head(10)\
                               .reset_index(drop=True)

    category_names = ['Retweets', 'Replies', 'Quotes', 'Likes']
    labels = df_top['tweet_id'].astype(str) + '\n(' + \
        df_top['interaction_count'].astype(str) + ')'

    max_iter = df_top['interaction_count'].max()
    data = df_top[['retweet_count', 'reply_count', 'quote_count',
                   'like_count']]

    data_cum = data.cumsum(axis=1)
    category_colors = plt.colormaps['RdYlGn'](
            np.linspace(0.15, 0.87, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data.iloc[:, i]
        starts = data_cum.iloc[:, i] - widths
        rects = ax.barh(labels, widths.to_numpy(), left=starts.to_numpy(),
                        height=0.7, label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.35 else 'black'
        proportion = widths / max_iter
        widths = [widths[k] if proportion[k] > 0.02 else '-'
                  if proportion[k] > 0.001 else '' for k in range(len(widths))]
        ax.bar_label(rects, label_type='center', color=text_color,
                     fontsize=22, labels=widths)

    ax.legend(ncols=len(category_names), loc='lower center', fontsize=28,
              bbox_to_anchor=(0.5, -0.025))
    plt.yticks(fontsize=22)
    for i, spine in enumerate(plt.gca().spines.values()):
        if i != 0:
            spine.set_visible(False)
    plt.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.03)

    fig.set_size_inches(32, 18)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f'images/top_posts/{metric}', exist_ok=True)
    plt.savefig(f'images/top_posts/{metric}/{date}.png', dpi=350)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a pie chart and ' +
                                     'circles plots with the proportions ' +
                                     'of the campaigns according to the ' +
                                     'number of uses')
    parser.add_argument('-i', '--ignore', action='store_true', default=False,
                        help='Flag to ignore the data in the data folder' +
                        ' and generate it from the database')
    args = parser.parse_args()

    db = DB(db_uri)
    os.makedirs('data', exist_ok=True)
    if 'tw_interactions.csv' not in os.listdir('data') or args.ignore:
        df_tw_interactions = db.get_tweets_metrics()
        df_tw_interactions.to_csv('data/tw_interactions.csv', index=False)
    else:
        df_tw_interactions = pd.read_csv('data/tw_interactions.csv')

    bar_comparation('retweet_count')

    db.close_connection()
