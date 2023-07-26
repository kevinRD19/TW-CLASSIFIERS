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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera un gráfico de ' +
                                     'con las proporciones de las ' +
                                     'campañas según el número de usos')
    parser.add_argument('-n', '--numcategories', type=int, default=100,
                        help='Indicates the top number of hashtags to show')
    parser.add_argument('-i', '--ignore', action='store_true', default=False,
                        help='Flag to ignore the data in the data folder' +
                        ' and generate a new one from the database')
    args = parser.parse_args()
    os.makedirs('data', exist_ok=True)
    if 'count_categories.csv' not in os.listdir('data') or args.ignore:
        db = DB(db_uri)
        df_count_cat = db.get_most_categories()
        df_count_cat.to_csv('data/count_categories.csv', index=False)
    else:
        df_count_cat = pd.read_csv('data/count_categories.csv')
    df_top_count = df_count_cat[:args.numcategories]

    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots()
    bar_plot = df_top_count.plot.bar(x='category', y='num_uses',
                                     ax=axes, legend=False,
                                     xlabel='', ylabel='',
                                     width=0.92 if args.numcategories <= 60
                                     else 0.77)

    for c in axes.containers:
        labels = df_top_count['num_uses']
        labels = [
            f'{count}' if int(count) < 10**6 else f'{int(count)/10**6:.2f}M'
            for count in labels
        ]
        axes.bar_label(c, labels=labels, label_type='edge', rotation=90,
                       fontsize=18, padding=4)

    axes.set_xticklabels(axes.get_xticklabels(), rotation=45, ha='right',
                         fontsize=20)
    axes.set_yticklabels([
        str(count) if count < 10**6 else f'{count/10**6:.2f}M'
        for count in np.arange(0, 2*10**6 + 1, 250000)],
                         fontsize=22)

    axes.get_yaxis().get_offset_text().set_position((-0.02, 0))
    for i, spine in enumerate(plt.gca().spines.values()):
        if i % 2:
            spine.set_visible(False)

    plt.subplots_adjust(left=0.05, right=1.0, top=0.98, bottom=0.19)
    fig = plt.gcf()
    fig.set_size_inches(32, 18)

    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('images/top_categories/', exist_ok=True)
    plt.savefig(f'images/top_categories/{date}.png', dpi=350)

    plt.show()
