# from functools import reduce
import json
import os
import re
import time
from typing import Tuple
import emoji
import inquirer
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from utils.DB import DB
import matplotlib.pyplot as plt


dirs = ['data/tree', 'data/tree2', 'data/tree3', 'data/tree4', 'data/tree5']
CONFIG = json.load(open('src/config.json', 'r'))


class Radar(object):
    def __init__(self, figure, title, labels, rect=None):
        if rect is None:
            rect = [0.05, 0.05, 0.9, 0.9]
        self.n = len(title)
        self.angles = np.arange(0, 360, 360.0/self.n)

        self.axes = [figure.add_axes(rect, projection='polar',
                                     label='axes%d' % i)
                     for i in range(self.n)]

        self.ax = self.axes[0]
        lines, ticks = self.ax.set_thetagrids(self.angles, labels=title,
                                              fontsize=22, fontweight='bold')
        num_axes = len(self.angles)
        for angle, tick in zip(self.angles, ticks):
            if num_axes == 4:
                if angle == 0:
                    tick.set_horizontalalignment('left')
                    tick.set_verticalalignment('top')
                    # tick.set_position((0, 0))
                elif angle == 90:
                    tick.set_horizontalalignment('center')
                    tick.set_verticalalignment('bottom')
                    # tick.set_position((0, 0))
                elif angle == 180:
                    tick.set_horizontalalignment('right')
                    tick.set_verticalalignment('bottom')
                    # tick.set_position((0, 0))
                elif angle == 270:
                    tick.set_horizontalalignment('center')
                    tick.set_verticalalignment('top')
                    # tick.set_position((0, 0))
            elif num_axes == 5:
                if angle == 0:
                    tick.set_horizontalalignment('left')
                    tick.set_verticalalignment('top')
                    # tick.set_position((0, 0))
                elif angle == 72:
                    tick.set_horizontalalignment('left')
                    tick.set_verticalalignment('bottom')
                    # tick.set_position((0, 0))
                elif angle == 144:
                    tick.set_horizontalalignment('right')
                    tick.set_verticalalignment('bottom')
                    # tick.set_position((0, 0))
                elif angle == 216:
                    tick.set_horizontalalignment('right')
                    tick.set_verticalalignment('top')
                    # tick.set_position((0, 0))
                else:
                    tick.set_horizontalalignment('left')
                    tick.set_verticalalignment('center')
                    # tick.set_position((0, 0))

        for ax in self.axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)

        for ax, angle, label in zip(self.axes, self.angles, labels):
            ax.set_rgrids(range(1, 6), angle=angle, labels=label)
            ax.spines['polar'].set_visible(False)
            ax.tick_params(axis='y', labelsize=20)
            ax.set_ylim(0, 5)

    def plot(self, values, *args, **kw):
        angle = np.deg2rad(np.r_[self.angles, self.angles[0]])
        values = np.r_[values, values[0]]
        self.ax.plot(angle, values, *args, **kw)


def clear_text(text: str, hashtag: bool = True) -> str:
    """
    Clear quotes from a text.

    Arguments
    ----------
        - text (`str`): text to be cleaned.

    Returns
    ----------
        `str`: text without quotes.
    """
    # text = _clear_retweet(text)
    text = clear_emoticons(text)
    text = clear_mentions(text)
    text = clear_url(text)
    if hashtag:
        text = clear_hashtags(text)
    text = clear_characters(text)
    # text = _clear_punctuation(text)
    # text = _clear_stopwords(text)
    # text = _clear_spaces(text)
    # print(text)
    return text


def clear_replies(text: str) -> str:
    """
    Clear quotes from a text.

    Arguments
    ----------
        - text (`str`): text to be cleaned.

    Returns
    ----------
        `str`: text without quotes.
    """
    i = 0
    tweets_text = text.split()
    for i, word in enumerate(tweets_text):
        if not word.startswith('@'):
            break
    return ' '.join(tweets_text[i:])


def clear_quote(text: str) -> str:
    """
    Clear quotes from a text.

    Arguments
    ----------
        - text (`str`): text to be cleaned.

    Returns
    ----------
        `str`: text without quotes.
    """
    return ' '.join(text.split()[:-1])


def clear_mentions(text: str) -> str:
    """
    Clear mentions from a text.

    Arguments
    ----------
        - text (`str`): text to be cleaned.

    Returns
    ----------
        `str`: text without quotes.
    """
    # text = clear_replies(text)
    text_words = text.split()
    text = ' '.join(word for word in text_words if not word.startswith('@'))
    # text = re.sub(r'^@*', '', text)
    return text


# def _clear_retweet(text: str) -> str:
#     """
#     Clear retweet from a text.

#     Arguments
#     ----------
#         - text (`str`): text to be cleaned.

#     Returns
#     ----------
#         `str`: text without retweet.
#     """
#     if text.startswith('RT @'):
#         text = text.split(':', 1)[1]
#     return text


def clear_url(text: str) -> str:
    """
    Clear urls from a text.

    Arguments
    ----------
        - text (`str`): text to be cleaned.

    Returns
    ----------
        `str`: text without urls.
    """
    pattern = "https?://[^\\s]+"
    return re.sub(pattern, '', text)


def clear_emoticons(text: str) -> str:
    """
    Obtains emoticons from a text.

    Args:
        text (str): _description_

    Returns:
        str: _description_
    """
    emotes = []
    words = []
    for word in text.split():
        word_emotes = [letter for letter in word if emoji.is_emoji(letter)] if\
            len(word) > 1 else [word] if emoji.is_emoji(word) else []
        # if word_emotes:
        #     print(word_emotes)
        for emote in word_emotes:
            word = word.replace(emote, '')
        if not emoji.is_emoji(word):
            words.append(word)
        else:
            emotes.append(word)
        emotes.extend(word_emotes)
    # print(' '.join(words))
    if len(words) >= 1:
        words[-1] = words[-1][:-2] if words[-1].endswith('…') else words[-1]
    return ' '.join(words)


def clear_hashtags(text: str) -> str:
    """
    Clear hashtags from a text.

    Arguments
    ----------
        - text (`str`): text to be cleaned.

    Returns
    ----------
        `str`: text without hashtags.
    """
    tweets_text = text.split()
    text = ''
    for i, word in enumerate(tweets_text):
        text += f'{word} ' if not word.startswith('#') and \
            i != len(tweets_text)-1\
            else f'{word}' if not word.startswith('#') else ''

    return text


def clear_characters(text: str) -> str:
    """
    Clear characters from a text.

    Arguments
    ----------
        - text (`str`): text to be cleaned.

    Returns
    ----------
        `str`: text without characters.
    """
    # pattern = "[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ]"
    # return re.sub(pattern, ' ', text)
    text = re.sub(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+|\s+$', '', text)
    # print(text)
    return text


def select_emotion_classifier():
    """
    Selects the emotion classifier.

    Returns
    ----------
        `str`: emotion classifier.
    """
    title = 'Select emotion classifier'
    options = ['BIPOLAR', 'MULTICLASS']
    response = inquirer.prompt([inquirer.List('emotion_classifier',
                                              message=title,
                                              choices=options)])
    return response['emotion_classifier']


def select_layout():
    title = 'Select layout for the graph'
    options = ['KAMADA_KAWAI', 'SPRING']
    response = inquirer.prompt([inquirer.List('layout', message=title,
                                              choices=options)])
    return f"{response['layout'].lower()}_layout"


def tweets(db, ignore) -> pd.DataFrame:
    """
    Get the tweets to generate the tree

    Returns
    -------
        : _description_
    """
    os.makedirs('data', exist_ok=True)
    if 'tweets_conexion.csv' not in os.listdir('data') or ignore:
        df_tweets = db.get_tweets_to_conexion()
        df_count = df_tweets.groupby(by=['conversation_id']).size()\
                            .reset_index(name='num_tweets')
        df_tweets = df_tweets.merge(df_count, how='left', on='conversation_id')
        df_tweets.to_csv('data/tweets_conexion.csv', index=False)
    else:
        df_tweets = pd.read_csv('data/tweets_conexion.csv')

    return df_tweets


def load_root_tweets(db, tweets: pd.DataFrame = None,
                     ignore: bool = False) -> pd.DataFrame:
    """
    Get the root tweets to generate the tree

    Returns
    -------
        : _description_
    """
    if os.path.exists('data/root_tweets.csv') and not ignore:
        df_root_tweets = pd.read_csv('data/root_tweets.csv')
    else:
        df_root_tweets = tweets.loc[
                             tweets['conversation_id'] == tweets['tweet_id']
                         ]
        df_root_tweets = db.get_root_tweets(df_root_tweets)
        df_root_tweets.to_csv('data/root_tweets.csv', index=False)

    return df_root_tweets


def load_tw_inter(db, ignore) -> pd.DataFrame:
    """
    Get the tweet metrics to generate the tree

    Returns
    -------
        : _description_
    """
    if 'tw_interactions.csv' not in os.listdir('data') or ignore:
        df_tw_interactions = db.get_tweets_metrics()
        df_tw_interactions.to_csv('data/tw_interactions.csv', index=False)
    else:
        df_tw_interactions = pd.read_csv('data/tw_interactions.csv')

    df_tw_interactions.drop(columns=['id'], inplace=True)
    df_tw_interactions.rename(columns={'tweet_id': 'id'}, inplace=True)

    return df_tw_interactions


def quotes(db):
    """_summary_

    Returns:
        _type_: _description_
    """
    qt = db.get_quotes()
    qt['quoted'] = qt['quoted'].astype(int)

    return list(set(qt['quoted'].to_list()))


def get_path(tweet: pd.Series):
    _file = f"{tweet['conversation_id']}.json"
    for _dir in dirs:
        if _file in os.listdir(_dir):
            print(_dir)
            return os.path.join(_dir, _file)


def get_useful_roots(db: DB, ignore: bool = False,
                     relevants: bool = False) -> Tuple[pd.DataFrame,
                                                       pd.DataFrame]:
    os.makedirs('data', exist_ok=True)
    df_tweets = tweets(db, ignore)
    if 'useful_roots.csv' not in os.listdir('data') or ignore:
        df_root_tweets = load_root_tweets(db, df_tweets, ignore)
        df_tw_inter = load_tw_inter(db, ignore)

        df_root_tweets = df_root_tweets.merge(df_tw_inter, how='left',
                                              on='id')

        df_root_tweets = df_root_tweets.loc[df_root_tweets['lang'] == 'es']. \
            copy()

        qt = quotes(db)
        df_root_tweets = df_root_tweets.loc[
            (df_root_tweets['num_tweets'] > 1) |
            (df_root_tweets['conversation_id'].isin(qt))
        ].copy() if relevants else df_root_tweets

        df_root_tweets.sort_values(by=['num_tweets', 'quote_count',
                                       'reply_count'],
                                   inplace=True, ascending=False)
        df_root_tweets.reset_index(drop=True, inplace=True)
        df_root_tweets.to_csv('data/useful_roots.csv', index=False)
    else:
        df_root_tweets = pd.read_csv('data/useful_roots.csv')

    return df_root_tweets, df_tweets


def load_tree(db, tweet: pd.Series, df_tweets: pd.DataFrame,
              dir: str = 'tree') -> Tuple[dict, pd.DataFrame]:
    os.makedirs(f'data/{dir}', exist_ok=True)
    path = get_path(tweet)
    if not path:
        path = f'data/{dir}/{tweet["conversation_id"]}.json'
        print(f'Generando árbol de conversación: {tweet["conversation_id"]}')
        start = time.time()
        tree, df_tweets = db.get_tree(df_tweets, tweet, True)
        end = time.time()
        with open(path, 'w') as outfile:
            print('Guardando árbol generado de conversación: ' +
                  f'{tweet["conversation_id"]}. Duración: ' +
                  f'{time.strftime("%H:%M:%S",time.gmtime(end-start))}')
            json.dump(tree, outfile, indent=4)
    else:
        with open(path, 'r') as infile:
            tree = json.load(infile)
    return tree, df_tweets
    # path = f'data/tree/{tweet["conversation_id"]}.json'
    # if f'{tweet["conversation_id"]}.json' not in os.listdir('data/tree'):
    #     tree, df_tweets = db.get_tree(df_tweets, tweet, True)
    #     with open(path, 'w') as outfile:
    #         print(f'Guardando árbol generado de conversación: {path}')
    #         json.dump(tree, outfile, indent=4)
    # else:
    #     with open(path, 'r') as infile:
    #         tree = json.load(infile)
    # return tree


def get_dates(db: DB, ignore: bool = False):
    if 'tweets_date_detail.csv' not in os.listdir('data') or ignore:
        df_posts_user = db.get_tweet_dates_details()
        df_posts_user.to_csv('data/tweets_date_detail.csv', index=False)
    else:
        df_posts_user = pd.read_csv('data/tweets_date_detail.csv')

    return df_posts_user


def get_tweet_text(db: DB, tweets: pd.DataFrame = None, ignore: bool = False,
                   clear: bool = False) -> pd.DataFrame:

    if 'tweet_text.pkl' not in os.listdir('data') or ignore:
        print('Getting tweet text')
        df_text = db.get_tweet_text()
        df_text.to_pickle('data/tweet_text.pkl', index=False)
    else:
        df_text = pd.read_pickle('data/tweet_text.pkl')

    if clear:
        if 'text_temp.pkl' not in os.listdir('data') or ignore:
            df_text = df_text[df_text['lang'].isin(['es', 'und'])]
            df_text = clear_tweets_text(df_text, db)
            df_text.to_pickle('data/text_temp.pkl')
        else:
            df_text = pd.read_pickle('data/text_temp.pkl')

    return df_text.loc[df_text['id'].isin(tweets['id'])] \
        if tweets is not None else df_text


def clear_tweets_text(df_text: pd.DataFrame, db: DB) -> pd.DataFrame:
    df_rtweet = db.get_retweets()
    df_text = df_text.loc[~df_text['id'].isin(df_rtweet['tweet_id'])]

    df_quotes = db.get_quotes()
    df_text['text'] = np.where(df_text['id'].isin(df_quotes['tweet_id']),
                               df_text['text'].apply(lambda x: clear_quote(x)),
                               df_text['text'])

    df_replies = db.get_replies()
    df_text['text'] = np.where(df_text['id'].isin(df_replies['tweet_id']),
                               df_text['text'].apply(
                                   lambda x: clear_replies(x)),
                               df_text['text'])

    return df_text


def save_plot(fig, name: str, dir: str = 'plots'):
    os.makedirs(f'{dir}', exist_ok=True)
    fig.savefig(f'{dir}/{name}.png', bbox_inches='tight', dpi=350)


def scatter_plot(df_tweets: pd.DataFrame, names: dict,
                 path: str = 'images/', name: str = 'scatter_plot'):
    tweets = df_tweets.drop(columns=['author_id'])
    tweets.rename(columns=names, inplace=True)
    axes = scatter_matrix(tweets, diagonal='kde', color='blue')

    [
        plt.setp(item.yaxis.get_majorticklabels(), 'size', 20)
        for item in axes.ravel()
    ]
    [
        plt.setp(item.xaxis.get_majorticklabels(), 'size', 20)
        for item in axes.ravel()
    ]
    [plt.setp(item.yaxis.get_label(), 'size', 22) for item in axes.ravel()]
    [plt.setp(item.xaxis.get_label(), 'size', 22) for item in axes.ravel()]

    plt.subplots_adjust(left=0.04, right=0.99, top=0.99, bottom=0.08)
    fig = plt.gcf()
    fig.set_size_inches(32, 18)
    os.makedirs(path, exist_ok=True)
    plt.savefig(f'{path}{name}.png', dpi=350)


def show_or_save():
    title = 'Do you want to show, save the plot or both?'
    options = ['SHOW', 'SAVE', 'SHOW AND SAVE']
    response = inquirer.prompt([inquirer.List('option',
                                              message=title,
                                              choices=options)])
    return response['option']
