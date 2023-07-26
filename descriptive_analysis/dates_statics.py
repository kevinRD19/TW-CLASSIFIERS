import argparse
from datetime import datetime
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

c_dir = os.path.dirname(os.path.abspath(__file__))
p_dir = os.path.dirname(c_dir)
sys.path.append(p_dir)

from utils.utils import (get_dates, CONFIG, # noqa
                         show_or_save) # noqa
from utils.DB import DB # noqa


db_uri = CONFIG['uri']
colors = []


def monthly_plot():
    lines = []
    df_posts_user['created_at'] = list(map(lambda x: x.replace(hour=0,
                                                               minute=0,
                                                               second=0,
                                                               year=2020,
                                                               day=1),
                                           df_posts_user['created_at']))
    tweets = df_posts_user.groupby(by=['year', 'created_at']).size()\
        .reset_index(name='num_posts_per_month')
    tweets.sort_values(by=['year', 'created_at'], inplace=True)
    dates = pd.date_range(start='2020-01-01', end='2021-01-01',
                          freq='M')
    dates = pd.Series(map(lambda x: x.replace(day=1), dates))
    years = tweets['year'].unique()

    for year in years:
        df_crono = pd.DataFrame({'year': year, 'created_at': dates})
        df_crono = df_crono.merge(tweets[tweets['year'] == year],
                                  how='left', on=['year',
                                                  'created_at'])
        df_crono['num_posts_per_month'] = df_crono['num_posts_per_month']\
            .fillna(0)
        cron, = plt.plot(df_crono['created_at'],
                         df_crono['num_posts_per_month'],
                         label=year, linewidth=2, marker='x')
        lines.append(cron)

    axes = plt.gca()
    plt.ylim(0,)
    plt.xlim(dates.min(), dates.max())

    values = plt.gca().get_yticks()
    axes.set_yticklabels(['{:.0f}'.format(x) for x in values])
    axes.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    axes.grid(True, axis='y', which='both', alpha=0.5)
    axes.tick_params(axis='x', which='major', pad=15)

    plt.subplots_adjust(left=0.07, right=0.98, top=0.97, bottom=0.05)
    plt.legend(loc='upper left',
               title='Year', fontsize=32, shadow=True, draggable=True,
               title_fontproperties={'size': 40, 'weight': 'bold'}
               )

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    fig = plt.gcf()
    fig.set_size_inches(32, 18)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('images/posts_by_year', exist_ok=True)
    plt.savefig(f'images/posts_by_year/{date}.png', dpi=300)
    plt.show()


def monthlydays_plot():
    lines = []
    tweets = df_posts_user.groupby(by=['year', 'created_at']).size()\
        .reset_index(name='num_posts_per_month')
    tweets.sort_values(by=['year', 'created_at'], inplace=True)
    tweets['days'] = list(map(lambda x:
                          x.replace(hour=0, minute=0, second=0,
                                    year=2020),
                          tweets['created_at']))
    dates = pd.date_range(start='2020-01-01', end='2020-12-01',
                          freq='D')
    colors = ['b', 'g', 'r', 'orange', 'purple', 'y', 'k', 'orange', 'pink',
              'm', 'c']
    years = tweets['year'].unique()

    for year, color in zip(years, colors[:len(years)]):
        df_crono = pd.DataFrame({'year': year, 'days': dates})
        df_crono = df_crono.merge(tweets[tweets['year'] == year],
                                  how='left', on=['year',
                                                  'days'])
        df_crono['num_posts_per_month'] = df_crono['num_posts_per_month']\
            .fillna(0)
        cron, = plt.plot(df_crono['days'], df_crono['num_posts_per_month'],
                         label=year, linewidth=0.5, color=color)
        lines.append(cron)

    axes = plt.gca()
    plt.ylim(0,)
    plt.xlim(dates.min(), dates.max())
    axes.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    axes.grid(True, axis='y', which='both', alpha=0.5)
    plt.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.06)
    plt.legend(loc='upper left',
               title='Year', title_fontsize=15, fontsize=12,
               shadow=True, draggable=True,
               )
    plt.show()


def time_plot():
    tweets = df_posts_user.groupby(by=['time']).size()\
        .reset_index(name='num_posts_per_hour')
    tweets.sort_values(by=['time'], inplace=True)
    hours = pd.date_range(start='2020-01-01', end='2020-01-02',
                          freq='H')[:-1]

    df_temp = pd.DataFrame({'time': hours})
    df_temp['time'] = df_temp['time'].dt.time
    df_temp = df_temp.merge(tweets, how='left', on=['time'])
    df_temp['time'] = df_temp['time'].fillna(0)

    df_temp.plot(x='time', y='num_posts_per_hour', kind='bar', legend=False)
    axes = plt.gca()
    for i, spine in enumerate(axes.spines.values()):
        if i % 2:
            spine.set_visible(False)
    axes.grid(True, axis='y', which='both', alpha=0.5)
    plt.xticks(rotation=0)
    plt.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.06)
    plt.show()


def stacked_time_plot():
    tweets = df_posts_user.groupby(by=['year', 'time']).size()\
        .reset_index(name='num_posts_per_hour')
    # tweets.sort_values(by=['year', 'time'], inplace=True)
    hours = pd.date_range(start='2020-01-01', end='2020-01-02',
                          freq='H')[:-1]
    years = list(tweets['year'].unique())
    rep_years = years*24
    df_aux = pd.DataFrame({'year': rep_years,
                           'time': np.repeat(hours, len(years))
                           })
    df_aux['time'] = df_aux['time'].dt.time
    df_aux = df_aux.merge(tweets, how='left', on=['year', 'time'])
    df_aux.sort_values(by=['year', 'time'], inplace=True)
    df_temp = pd.DataFrame({year: df_aux.loc[df_aux['year'] == year,
                                             'num_posts_per_hour'].
                            reset_index(drop=True) for year in years})
    df_temp.fillna(0, inplace=True)
    df_temp = df_temp.astype(int)
    df_temp.set_index(df_aux['time'].unique(), inplace=True)

    df_temp.plot.bar(rot=0, stacked=True, zorder=5, alpha=0.8, width=0.85)
    plt.legend(loc='upper center', ncol=len(df_temp.columns),
               fontsize=30, shadow=True, draggable=True,
               )

    axes = plt.gca()
    axes.set_xticklabels([x.strftime("%H:%M") for x in df_temp.index])
    plt.xticks(fontsize=27)
    plt.yticks(fontsize=30)
    # axes.bar_label(bar_plot, label_type='center', zorder=10)
    heights = [[int(v.get_height()) for v in c] for c in axes.containers]
    max_height = max([sum(height_c) for height_c in heights])
    for i, c in enumerate(axes.containers):
        labels = [height if height/max_height > 0.003 else '-'
                  for height in heights[i]]
        axes.bar_label(c, labels=labels, label_type='center', zorder=10,
                       fontsize=22)

    for i, spine in enumerate(axes.spines.values()):
        if i % 2:
            spine.set_visible(False)
    axes.grid(True, axis='y', which='both', alpha=0.5, zorder=0)

    fig = plt.gcf()
    fig.set_size_inches(32, 18)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.subplots_adjust(left=0.06, right=1.0, top=0.97, bottom=0.05)
    os.makedirs('images/time_preferences', exist_ok=True)
    plt.savefig(f'images/time_preferences/{date}.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera un gráfico de ' +
                                     'con las proporciones de las ' +
                                     'campañas según el número de usos')
    parser.add_argument('-i', '--ignore', action='store_true', default=False,
                        help='Flag to ignore the data in the data folder' +
                        ' and generate a new one from the database')
    args = parser.parse_args()
    os.makedirs('data', exist_ok=True)
    db = DB(db_uri)
    df_posts_user = get_dates(db, args.ignore)

    df_posts_user['created_at'] = pd.to_datetime(df_posts_user['created_at'])
    df_posts_user['year'] = df_posts_user['created_at'].dt.year
    df_posts_user['time'] = list(map(lambda x: x.replace(minute=0, second=0),
                                     df_posts_user['created_at'].dt.time))

    df_posts_user.sort_values(by=['year', 'time'], inplace=True)

    plt.rcParams["font.family"] = "serif"
    # monthlydays_plot()
    # monthly_plot()
    # time_plot()
    stacked_time_plot()

    db.close_connection()
