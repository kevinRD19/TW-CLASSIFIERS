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

from utils.utils import CONFIG, show_or_save # noqa
from utils.DB import DB # noqa

db_uri = CONFIG['uri']


def top_mentions():
    """
    Generates a bar plot of the top n mentioned names.
    """
    fig, axes = plt.subplots()
    df_top_count.plot.bar(x='name', y='num_uses',
                          ax=axes, legend=False,
                          xlabel='', ylabel='',
                          width=0.9 if args.numnames <= 50 else 0.8)
    for container in axes.containers:
        axes.bar_label(container, label_type='edge', rotation=90,
                       fontsize=18, padding=4)

    axes.set_xticklabels(axes.get_xticklabels(), rotation=45, ha='right')
    axes.xaxis.set_tick_params(labelsize=19, pad=0)
    plt.yticks(fontsize=23)
    for i, spine in enumerate(plt.gca().spines.values()):
        if i % 2:
            spine.set_visible(False)

    plt.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.17)
    fig.set_size_inches(32, 18)

    show_or_save(plt, 'images/mentions/top')


def temporal_evolution():
    """
    Generates a temporal evolution plot of the top n mentioned names.
    """
    if 'name_temp.csv' not in os.listdir('data') or args.ignore:
        tweets = db.get_tweets_by_mention(
            df_top_count['name'][:args.temporal]
        )
        tweets['created_at'] = list(map(lambda x:
                                    x.replace(hour=0, minute=0,
                                              second=0, day=1),
                                    tweets['created_at']))
        tweets = tweets.groupby(by=['name', 'created_at']).size()\
            .reset_index(name='num_uses_per_month')
        tweets.sort_values(by=['name', 'created_at'], inplace=True)
        tweets.to_csv('data/name_temp.csv', index=False)
    else:
        tweets = pd.read_csv('data/name_temp.csv')
        tweets['created_at'] = pd.to_datetime(tweets['created_at'])
    start = tweets['created_at'].min()
    end = tweets['created_at'].max()
    dates = pd.date_range(start=start, end=end, freq='M')
    dates = list(map(lambda x: x.replace(hour=0, minute=0, second=0, day=1),
                     dates))
    dates = pd.to_datetime(dates)
    for name in tweets['name'].unique()[:args.temporal]:
        df_temp = pd.DataFrame({'name': name, 'created_at': dates})
        df_temp = df_temp.merge(tweets[tweets['name'] == name],
                                how='left', on=['name',
                                                'created_at'])
        df_temp['num_uses_per_month'] = df_temp['num_uses_per_month']\
            .fillna(0)
        plt.plot(df_temp['created_at'], df_temp['num_uses_per_month'],
                 label=name, marker='x', linewidth=2.5)
    axes = plt.gca()
    plt.ylim(0,)
    plt.xlim(start, end)
    ystart, yend = axes.get_ylim()
    plt.yticks(np.arange(ystart, yend + 1, 20000))
    axes.yaxis.set_minor_locator(MultipleLocator(1000))
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
               title='Names', fontsize=25,
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

    show_or_save(plt, 'images/mentions/time_evolution')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Genera un gráfico de ' +
                                     'con las proporciones de las ' +
                                     'campañas según el número de usos')
    parser.add_argument('-n', '--numnames', type=int, default=100,
                        help='Indicates the top number of mentioned names ' +
                        'to show')
    parser.add_argument('-t', '--temporal', type=int, default=10,
                        help='Indicates the number of mentioned names ' +
                        'to show in the temporal evolution plot')
    parser.add_argument('-i', '--ignore', action='store_true', default=False,
                        help='Flag to ignore the data in the data folder' +
                        ' and generate it from the database')
    args = parser.parse_args()

    db = DB(db_uri)
    os.makedirs('data', exist_ok=True)
    if 'count_quoted_names.csv' not in os.listdir('data') or args.ignore:
        df_count_name = db.get_most_person()
        df_count_name.to_csv('data/count_quoted_names.csv', index=False)
    else:
        df_count_name = pd.read_csv('data/count_quoted_names.csv')
    df_top_count = df_count_name[:args.numnames]

    top_mentions()
    temporal_evolution()

    db.close_connection()
