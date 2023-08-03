import argparse
import os
import sys
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

c_dir = os.path.dirname(os.path.abspath(__file__))
p_dir = os.path.dirname(c_dir)
sys.path.append(p_dir)

from utils.utils import (clean_quote, clean_replies, # noqa
                         show_or_save, CONFIG) # noqa
from utils.DB import DB # noqa


db_uri = CONFIG['uri']


def lenght_statistic():
    if 'tweet_text.pkl' not in os.listdir('data') or args.ignore:
        df_tweet = db.get_tweet_text()
        df_tweet.to_pickle('data/tweet_text.pkl')
    else:
        df_tweet = pd.read_pickle('data/tweet_text.pkl')

    df_rtweet = db.get_retweets()
    df_tweet = df_tweet.loc[~df_tweet['id'].isin(df_rtweet['tweet_id'])]

    df_quotes = db.get_quotes()
    df_tweet['text'] = np.where(df_tweet['id'].isin(df_quotes['tweet_id']),
                                df_tweet['text'].apply(
                                    lambda x: clean_quote(x)),
                                df_tweet['text'])

    df_replies = db.get_replies()
    df_tweet['text'] = np.where(df_tweet['id'].isin(df_replies['tweet_id']),
                                df_tweet['text'].apply(
                                    lambda x: clean_replies(x)),
                                df_tweet['text'])

    df_tweet['Number of characters'] = df_tweet['text'].apply(lambda x: len(x))
    df_statistics = df_tweet.groupby(by=['author_id'])['Number of characters']\
                            .mean() \
                            .apply(lambda x: round(x, 0)).replace(np.nan, 0) \
                            .reset_index(name='Mean')

    bins = int(round(df_statistics['Mean'].max(), -1) / 10)
    df_statistics.hist(column='Mean', bins=bins*2, alpha=0.8, figsize=(20, 10),
                       rwidth=0.75, grid=False, density=True, zorder=10)
    df_statistics['Mean'].plot.kde(figsize=(20, 10), grid=False, zorder=15,
                                   alpha=0.9, linewidth=3)

    axes = plt.gca()
    axes.set_xlim([0, (bins)*10])
    ymax = axes.get_ylim()[1]
    axes.xaxis.set_major_locator(MultipleLocator(50))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    axes.axvline(df_statistics['Mean'].quantile(0.25), color='red', alpha=0.8,
                 linestyle="dashed", zorder=5)
    axes.axvline(df_statistics['Mean'].quantile(0.50), color='red', alpha=0.8,
                 linestyle="dashed", zorder=5)
    axes.axvline(df_statistics['Mean'].quantile(0.75), color='red', alpha=0.8,
                 linestyle="dashed", zorder=5)

    max_ = int(df_statistics['Mean'].max())
    min_ = int(df_statistics['Mean'].min())
    mean_ = int(df_statistics['Mean'].mean())
    axes.axvline(min_, color='black', alpha=0.8,
                 linestyle="dashed", zorder=5, linewidth=3)
    axes.text(min_, ymax, f"MIN: {min_}", size=16, alpha=1,
              horizontalalignment='center', bbox={'edgecolor': 'black',
                                                  'boxstyle': 'round',
                                                  'facecolor': 'white'},
              zorder=20)
    axes.axvline(max_, color='black', alpha=0.8,
                 linestyle="dashed", zorder=5, linewidth=3)
    axes.text(max_, ymax, f"MAX: {max_}", size=16, alpha=1,
              horizontalalignment='center', bbox={'edgecolor': 'black',
                                                  'boxstyle': 'round',
                                                  'facecolor': 'white'},
              zorder=20)
    axes.axvline(mean_, color='black', alpha=0.8,
                 linestyle="dashed", zorder=5, linewidth=3)
    axes.text(mean_, ymax*0.8, f"MEAN: {mean_}", size=16, alpha=1,
              horizontalalignment='center', bbox={'edgecolor': 'black',
                                                  'boxstyle': 'round',
                                                  'facecolor': 'white'},
              zorder=20)

    for i, spine in enumerate(plt.gca().spines.values()):
        if i != 2:
            spine.set_visible(False)

    axes.tick_params(which='both', left=False, bottom=False)
    fig = plt.gcf()
    fig.set_size_inches(32, 18)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.04)
    axes.set_title('')
    plt.xlabel('')
    plt.ylabel('')

    path = 'images/len_stats'
    show_or_save(plt, path)


def hashtag_usage_statistic():
    if 'ht_statistics.csv' not in os.listdir('data') or args.ignore:
        df_statistics = db.hashtag_usage_statistics()
        df_statistics.to_csv('data/ht_statistics.csv', index=False)
    else:
        df_statistics = pd.read_csv('data/ht_statistics.csv')

    df_statistics = df_statistics.groupby(by=['author_id'])['used_hashtags']\
                                 .mean().apply(lambda x: round(x, 0))\
                                 .replace(np.nan, 0).reset_index(name='Mean')
    df_users = db.get_users()
    df_statistics = df_users.merge(df_statistics, how='left', on='author_id')\
                            .replace(np.nan, 0)

    xmax = int(round(df_statistics['Mean'].max(), 0))
    df_statistics.hist(column='Mean', bins=xmax+1, alpha=0.8, figsize=(20, 10),
                       grid=False, density=True, zorder=10, rwidth=0.85)
    df_statistics['Mean'].plot.kde(grid=False, zorder=15,
                                   alpha=0.9, linewidth=3)

    axes = plt.gca()
    axes.set_xlim([0, xmax])
    ymax = axes.get_ylim()[1]
    axes.xaxis.set_major_locator(MultipleLocator(2))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    axes.axvline(df_statistics['Mean'].quantile(0.25), color='red', alpha=0.8,
                 linestyle="dashed", zorder=5, linewidth=3)
    axes.axvline(df_statistics['Mean'].quantile(0.50), color='red', alpha=0.8,
                 linestyle="dashed", zorder=5, linewidth=3)
    axes.axvline(df_statistics['Mean'].quantile(0.75), color='red', alpha=0.8,
                 linestyle="dashed", zorder=5, linewidth=3)

    max_ = int(df_statistics['Mean'].max())
    min_ = int(df_statistics['Mean'].min())
    mean_ = int(df_statistics['Mean'].mean())
    axes.axvline(min_, color='black', alpha=0.8,
                 linestyle="dashed", zorder=5, linewidth=3)
    axes.text(min_, ymax, f"MIN: {min_}", size=16, alpha=1,
              horizontalalignment='center', bbox={'edgecolor': 'black',
                                                  'boxstyle': 'round',
                                                  'facecolor': 'white'},
              zorder=20)
    axes.axvline(max_, color='black', alpha=0.8,
                 linestyle="dashed", zorder=5, linewidth=3)
    axes.text(max_, ymax, f"MAX: {max_}", size=16, alpha=1,
              horizontalalignment='center', bbox={'edgecolor': 'black',
                                                  'boxstyle': 'round',
                                                  'facecolor': 'white'},
              zorder=20)
    axes.axvline(mean_, color='black', alpha=0.8,
                 linestyle="dashed", zorder=5, linewidth=3)
    axes.text(mean_, ymax*0.8, f"MEAN: {mean_}", size=16, alpha=1,
              horizontalalignment='center', bbox={'edgecolor': 'black',
                                                  'boxstyle': 'round',
                                                  'facecolor': 'white'},
              zorder=20)

    for i, spine in enumerate(plt.gca().spines.values()):
        if i != 2:
            spine.set_visible(False)

    axes.tick_params(which='both', left=False, bottom=False)
    fig = plt.gcf()
    fig.set_size_inches(32, 18)
    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.04)
    axes.set_title('')
    plt.xlabel('')
    plt.ylabel('')

    path = 'images/hashtag_stats'
    show_or_save(plt, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera un gráfico de ' +
                                     'con las proporciones de las ' +
                                     'campañas según el número de usos')
    parser.add_argument('-i', '--ignore', action='store_true', default=False,
                        help='Flag to ignore the data in the data folder' +
                        ' and generate it from the database')
    args = parser.parse_args()
    os.makedirs('data', exist_ok=True)
    db = DB(db_uri)

    lenght_statistic()
    hashtag_usage_statistic()

    db.close_connection()
