import argparse
from datetime import datetime
import circlify
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

c_dir = os.path.dirname(os.path.abspath(__file__))
p_dir = os.path.dirname(c_dir)
sys.path.append(p_dir)

from utils.DB import DB # noqa
from utils.utils import CONFIG, show_or_save # noqa


db_uri = CONFIG['uri']


def bubbles_graph():
    circles = circlify.circlify(df_count_campaign['num_uses'].tolist(),
                                show_enclosure=False,
                                target_enclosure=circlify.Circle(x=0, y=0)
                                )
    circles.reverse()
    df_count_campaign['percent'] =\
        df_count_campaign['num_uses']/df_count_campaign['num_uses'].sum()*100
    df_count_campaign['percent'] =\
        df_count_campaign['percent'].map(lambda x: round(x, 1))
    label = [row['campaign'] + '\n' +
             format(row['num_uses'], ',').replace(',', ' ') +
             '\n' + str(row['percent'])+'%'
             for i, row in df_count_campaign.iterrows()]
    fig, ax = plt.subplots(figsize=(14, 14), facecolor='white')
    ax.axis('off')
    lim = max(max(abs(circle.x)+circle.r, abs(circle.y)+circle.r,)
              for circle in circles)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    for circle, note, color in zip(circles, label, pal_):
        x, y, r = circle
        ax.add_patch(plt.Circle((x, y), r, alpha=1, color=color))
        if r > 0.16:
            plt.annotate(note, (x, y), size=28,
                         va='center', ha='center')
        elif r > 0.1:
            plt.annotate(note, (x, y), size=23,
                         va='center', ha='center', color='white')
        else:
            plt.annotate(note, (x, y), size=17,
                         va='center', ha='center', color='white')
    plt.subplots_adjust(left=0.1, right=0.9, top=1.03, bottom=-0.02)
    plt.xticks([])
    plt.yticks([])

    fig.set_size_inches(32, 18)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('images/campaign/bubble', exist_ok=True)
    plt.savefig(f'images/campaign/bubble/{date}.png', dpi=350)


def pie_plot():
    fig, axes = plt.subplots()
    pie_plot = df_count_campaign.plot.pie(y='num_uses', ax=axes,
                                          colors=pal_,
                                          autopct='%1.1f%%', startangle=90,
                                          pctdistance=0.5,
                                          labeldistance=1.1,
                                          wedgeprops={'linewidth': 2.0,
                                                      'edgecolor': 'white'},
                                          textprops={
                                              'size': '13',
                                              'weight': 'bold',
                                              'ha': 'center',
                                              'va': 'center',
                                              'bbox': dict(fc='white',
                                                           ec='black',
                                                           boxstyle='round,' +
                                                           'pad=0.2')
                                              },
                                          labels=None)
    plt.legend(df_count_campaign['campaign'] + '  (' +
               df_count_campaign['num_uses'].map(
               lambda x: format(x, ',').replace(',', ' ')) + ')',
               loc='center left',
               bbox_to_anchor=(1.0, 0.5), fontsize=12,
               title='Campañas',
               shadow=True,
               draggable=True,
               labelspacing=1.5,
               borderpad=1.5,
               title_fontproperties={'size': 14,
                                     'weight': 'bold',
                                     }
               )
    perc = 0.47/(num_camp//2)
    p = 0.5
    for i, text in enumerate(pie_plot.texts):
        if i % 2 and i//2 > num_camp//2:
            p += perc
            text.set_position((text.get_position()[0]*p/0.5,
                               text.get_position()[1]*p/0.5))
    axes.xaxis.set_tick_params(labelsize=8)
    axes.set_ylabel('')
    plt.subplots_adjust(left=0.01, right=0.99, top=1.12, bottom=-0.12)

    fig.set_size_inches(32, 18)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('images/campaign/pie', exist_ok=True)
    plt.savefig(f'images/campaign/pie/{date}.png', dpi=350)
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a pie chart and ' +
                                     'circles plots with the proportions of ' +
                                     'the campaigns according to the number ' +
                                     'of uses')
    parser.add_argument('-i', '--ignore', action='store_true', default=False,
                        help='Flag to ignore the data in the data folder' +
                        ' and generate it from the database')
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)
    if 'count_campaign_use.csv' not in os.listdir('data') or args.ignore:
        db = DB(db_uri)
        df_count_campaign = db.get_most_campaign()
        df_count_campaign.to_csv('data/count_campaign_use.csv', index=False)
    else:
        df_count_campaign = pd.read_csv('data/count_campaign_use.csv')

    plt.rcParams["font.family"] = "serif"
    num_camp = len(df_count_campaign)
    pal_ = list(sns.color_palette(palette='plasma_r',
                                  n_colors=num_camp))

    bubbles_graph()
    # pie_plot()

    plt.show()
