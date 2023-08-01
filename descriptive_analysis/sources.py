import argparse
from datetime import datetime
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

c_dir = os.path.dirname(os.path.abspath(__file__))
p_dir = os.path.dirname(c_dir)
sys.path.append(p_dir)

from utils.DB import DB # noqa
from utils.utils import CONFIG # noqa


db_uri = CONFIG['uri']


def pie_plot():
    fig, axes = plt.subplots()
    colors = ['#A4C639', '#A3AAAE', '#1DA1F2', 'red', 'purple', 'black']
    df_5sources = df_sources.head(5).copy()
    df_others = pd.DataFrame(data={
                    'source': ['Others'],
                    'num_tweets': [df_sources['num_tweets'][5:].sum()]
    })
    df_5sources = pd.concat([df_5sources,
                             df_others]).reset_index(drop=True)
    pie_plot = df_5sources.plot.pie(y='num_tweets', ax=axes, colors=colors,
                                    autopct='%1.1f%%', startangle=90,
                                    pctdistance=0.5, labeldistance=1.1,
                                    wedgeprops={'linewidth': 2.0,
                                                'edgecolor': 'white'},
                                    textprops={'size': '28',
                                               'weight': 'bold',
                                               'ha': 'center',
                                               'va': 'center',
                                               'bbox': dict(fc='white',
                                                            ec='black',
                                                            boxstyle='round,' +
                                                            'pad=0.4')},
                                    labels=None)
    plt.legend(df_5sources['source'] + '\n (' +
               df_5sources['num_tweets'].map(
               lambda x: format(x, ',')) + ')',
               loc='center left', bbox_to_anchor=(0.92, 0.5),
               fontsize=28, title='Platforms', shadow=True,
               draggable=True, labelspacing=1.5,
               borderpad=1.5,
               title_fontproperties={'size': 31,
                                     'weight': 'bold',
                                     }
               )

    p = (0.72, 0.6)
    for i, text in enumerate(pie_plot.texts):
        if i % 2 and text.get_text() == '1.9%':
            text.set_position((text.get_position()[0]*p[0]/0.5,
                               text.get_position()[1]*p[0]/0.5))
        else:
            text.set_position((text.get_position()[0]*p[1]/0.5,
                               text.get_position()[1]*p[1]/0.5))

    axes.set_ylabel('')
    plt.subplots_adjust(left=0.01, right=0.99, top=1.12, bottom=-0.12)

    fig.set_size_inches(32, 18)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('images/source', exist_ok=True)
    plt.savefig(f'images/source/{date}.png', dpi=350)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a pie chart ' +
                                     'with the proportions of the' +
                                     'sources according to the number of uses')
    parser.add_argument('-i', '--ignore', action='store_true', default=False,
                        help='Flag to ignore the data in the data folder' +
                        ' and generate it from the database')
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)
    db = DB(db_uri)

    if 'sources.csv' not in os.listdir('data') or args.ignore:
        df_sources = db.get_sources_count()
        df_sources.to_csv('data/sources.csv', index=False)
    else:
        df_sources = pd.read_csv('data/sources.csv')

    plt.rcParams["font.family"] = "serif"

    pie_plot()
    db.close_connection()
