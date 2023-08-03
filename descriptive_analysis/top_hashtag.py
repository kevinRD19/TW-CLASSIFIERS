import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

c_dir = os.path.dirname(os.path.abspath(__file__))
p_dir = os.path.dirname(c_dir)
sys.path.append(p_dir)

from utils.DB import DB # noqa
from utils.utils import CONFIG, show_or_save # noqa


db_uri = CONFIG['uri']


def temporal_evolution():
    """
    Generates a temporal plot of the top n hashtags.
    """
    if 'hashtag_temp.csv' not in os.listdir('data') or args.ignore:
        tweets = db.get_tweets_by_hashtag(
            df_count_hashtags['hashtag'][:args.temporal]
        )
        tweets['created_at'] = list(map(lambda x:
                                    x.replace(hour=0, minute=0,
                                              second=0, day=1),
                                    tweets['created_at']))
        tweets = tweets.groupby(by=['hashtag', 'created_at']).size()\
            .reset_index(name='num_uses_per_month')
        tweets.sort_values(by=['hashtag', 'created_at'], inplace=True)
        tweets.to_csv('data/hashtag_temp.csv', index=False)
    else:
        tweets = pd.read_csv('data/hashtag_temp.csv')
        tweets['created_at'] = pd.to_datetime(tweets['created_at'])
    start = tweets['created_at'].min()
    end = tweets['created_at'].max()
    dates = pd.date_range(start=start, end=end, freq='M')
    dates = list(map(lambda x: x.replace(hour=0, minute=0, second=0, day=1),
                     dates))
    dates = pd.to_datetime(dates)
    for hashtag in tweets['hashtag'].unique()[:args.temporal]:
        df_crono = pd.DataFrame({'hashtag': hashtag, 'created_at': dates})
        df_crono = df_crono.merge(tweets[tweets['hashtag'] == hashtag],
                                  how='left', on=['hashtag',
                                                  'created_at'])
        df_crono['num_uses_per_month'] = df_crono['num_uses_per_month']\
            .fillna(0)
        plt.plot(df_crono['created_at'], df_crono['num_uses_per_month'],
                 label=hashtag, marker='x', linewidth=2.5)
    axes = plt.gca()
    plt.ylim(0,)
    plt.xlim(start, end)
    ystart, yend = axes.get_ylim()
    plt.yticks(np.arange(ystart, yend + 1, 50000))
    axes.yaxis.set_minor_locator(MultipleLocator(25000))
    axes.xaxis.set_minor_locator(mdates.MonthLocator([1, 5, 9], interval=1))
    axes.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))

    axes.xaxis.set_major_locator(mdates.YearLocator(month=7))
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes.xaxis.set_tick_params(which='major', pad=30)
    axes.tick_params(which='major', tick1On=False)

    plt.setp(axes.get_xmajorticklabels(), rotation=0, weight="bold",
             ha="center", fontsize=22, va="top")
    axes.grid(True, axis='y', which='both', alpha=0.5)
    plt.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.06)
    plt.legend(loc='upper left',
               title='Hashtags', fontsize=25,
               shadow=True, draggable=True,
               title_fontproperties={'size': 28,
                                     'weight': 'bold',
                                     }
               )
    axes.tick_params(axis='both', which='minor', length=8, width=1,
                     labelsize=20)
    plt.yticks(fontsize=20)

    for i, spine in enumerate(axes.spines.values()):
        if i % 2:
            spine.set_visible(False)

    fig = plt.gcf()
    fig.set_size_inches(32, 18)

    show_or_save(plt, 'images/top_hashtag/time_evolution')


def top_hashtags():
    """
    Generates a bar plot with the top n hashtags according to the number
    of uses.
    """
    fig, axes = plt.subplots()
    width = 0.92 if args.numhashtags <= 50 else 0.8
    df_top_count.plot.bar(x='hashtag', y='num_uses', ax=axes,
                          legend=False, xlabel='', ylabel='',
                          width=width)
    if args.numhashtags <= 50:
        label_type = 'center'
        color = 'white'
        weight = 'demibold'
        pad = 0
        fsize = 15
    else:
        label_type = 'edge'
        color = 'black'
        weight = 'normal'
        pad = 4
        fsize = 8
    for container in axes.containers:
        axes.bar_label(container, label_type=label_type, rotation=90,
                       color=color, weight=weight, fontsize=fsize,
                       padding=pad)

    axes.set_xticklabels(axes.get_xticklabels(), rotation=45, ha='right')
    axes.xaxis.set_tick_params(labelsize=8, pad=0)
    for i, spine in enumerate(plt.gca().spines.values()):
        if i % 2:
            spine.set_visible(False)

    plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=20)

    fig = plt.gcf()
    fig.set_size_inches(32, 18)

    show_or_save(plt, 'images/top_hashtag/total_uses')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a bar plot ' +
                                     'with the top hashtags according to' +
                                     'the number of uses. Additionally, ' +
                                     'shows the temporal evolution of the' +
                                     'top n hashtags.')
    parser.add_argument('-n', '--numhashtags', type=int, default=100,
                        help='Indicates the top number of hashtags to show')
    parser.add_argument('-t', '--temporal', type=int, default=10,
                        help='Indicates the top number of hashtags to show' +
                        ' in the temporal plot')
    parser.add_argument('-i', '--ignore', action='store_true', default=False,
                        help='Flag to ignore the data in the data folder' +
                        ' and generate it from the database')
    args = parser.parse_args()

    db = DB(db_uri)
    os.makedirs('data', exist_ok=True)
    if 'count_hashtag_use.csv' not in os.listdir('data') or args.ignore:
        df_count_hashtags = db.get_most_hashtags()
        df_count_hashtags.to_csv('data/count_hashtag_use.csv', index=False)
    else:
        df_count_hashtags = pd.read_csv('data/count_hashtag_use.csv')
    df_top_count = df_count_hashtags[:args.numhashtags]

    top_hashtags()
    temporal_evolution()

    db.close_connection()
