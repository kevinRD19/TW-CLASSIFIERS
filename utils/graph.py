import networkx as nx
import matplotlib.pyplot as plt
import warnings
import numpy as np

import pandas as pd
from utils.node import Like, Quote, Reply, Retweet, Tweet
from utils.DB import DB
from utils.utils import (select_emotion_classifier, select_layout,
                         CONFIG, show_or_save)
from nrclex import NRCLex
from random import sample
from enum import Enum

warnings.filterwarnings("ignore")

# URI of the database with the appropiate credentials
db_uri = CONFIG['uri']

# Colors, sizes and weights of the nodes depending on the type of node
colors = {'Tweet': 'blue', 'Retweet': 'green', 'Like': 'red',
          'Quote': 'yellow', 'Reply': 'orange'}
sizes = {'Tweet': 5000, 'Retweet': 800, 'Like': 1000, 'Quote': 3000,
         'Reply': 3000}
weights = {'RT': 0.1, 'LK': 0.2, 'QT': 1, 'RP': 2}

# Layouts and their attributes
layouts_attr = {nx.spring_layout: {'k': 1}, nx.kamada_kawai_layout: {}}

# Colors for the emotions in the multiclass classifier
emot_colors = {'anger': 'red', 'anticipation': 'orange', 'joy': 'yellow',
               'trust': 'olive', 'fear': 'green', 'surprise': 'teal',
               'sadness': 'blue', 'disgust': 'violet', 'negative': 'black',
               'positive': '#D0D3D4', 'neutral': '#616A6B'}

# Colors for the emotions in the extended bipolar classifier
bipolar_colors = {'negative': 'red', 'neutral-negative': 'blue',
                  'neutral': 'black', 'neutral-positive': 'yellow',
                  'positive': 'green', }


class EmotionType(Enum):
    """
    Enumeration that represents the different types of tone classifier.
    """
    BIPOLAR = 0
    MULTICLASS = 1


class TweetGrah(nx.DiGraph):
    """
    Directed graph graph to represent the interactions of users
    with a tweet.
    """
    db = DB(db_uri)

    def __init__(self, graph_data: dict = None, rt: bool = True,
                 rp: bool = True, like: bool = True, qt: bool = True, **attr):
        """
        Class constructor.

        Arguments
        ----------
            - graph_data (dict, optional): Dictionary or JSON that contains
            a tweet interactions info. Defaults to None.
        """
        super().__init__(**attr)
        if graph_data:
            self.edge = {}
            self.quotes = qt
            self.retweets = rt
            self.replies = rp
            self.likes = like
            # Create the root node
            root = self.db.get_tweets_by_ids(graph_data['root'],
                                             ['id', 'author_id', 'text',
                                              'created_at']).iloc[0]
            self.start = root['created_at']
            self.root = Tweet(root)
            self.add_node(self.root)
            self._add_node_interactions(self.root, graph_data)

    def _add_node_interactions(self, parent: Tweet, graph_data: dict = {},
                               replies: bool = True):
        """
        Adds the interactions of a tweet to the graph.

        Arguments
        ----------
            - parent (`Tweet`): Tweet (node) of which we want to add the
            interactions.
            - graph_data (`dict`, `optional`): Dictionary with the tweet
            interactions. This dict works as a JSON Defaults to {}.
        """
        if self.retweets:
            retweets = graph_data['retweets']
            for retweet in retweets:
                self._retweet(parent, retweet)

        if self.likes:
            likes = graph_data['likes']
            for like in likes:
                self._like(parent, like)

        if self.quotes:
            quotes = graph_data['quotes']
            for quote in quotes:
                self._quotes(parent, quote)

        if self.replies and replies:
            replies = graph_data['replies']
            for reply in replies:
                self._replies(parent, reply)

    def _quotes(self, parent: Tweet, graph_data: dict = {}, level: int = 1):
        """
        Adds a quote node to the graph and its corresponding interactions
        recursively.

        Arguments
        ----------
            - parent (Tweet): Tweet (node) of which we want to add the quote.
            - graph_data (dict, optional): Dictionary with the quote
            interactions info. This dict works as a JSON. Defaults to {}.
            - level (int, optional): _description_. Defaults to 1.
        """
        id = int(graph_data['id'])
        tweet = self.db.get_tweets_by_ids(id, ['id', 'author_id', 'text',
                                               'created_at']).iloc[0]
        qt = Quote(tweet)
        self.add_node(qt)
        self.add_edge(parent, qt, weight=weights['QT'])
        self.edge[(parent, qt)] = 'QT'
        self._add_node_interactions(qt, graph_data)

    def _like(self, parent: Tweet, user_id: int):
        """
        Adds a like node to the graph.

        Arguments
        ----------
            - parent (Tweet): Tweet (node) of which we want to add the like.
            - user_id (int): Id of the user who liked the tweet.
        """
        like = Like(user_id)
        self.add_node(like)
        self.add_edge(parent, like, weight=weights['LK'])
        self.edge[(parent, like)] = 'LK'

    def _retweet(self, parent: Tweet, user_id: int):
        """
        Adds a retweet node to the graph.

        Arguments
        ----------
            - parent (Tweet): Tweet (node) of which we want to add the retweet.
            - user_id (int): Id of the user who did retweeted
        """
        rt = Retweet(user_id)
        self.add_node(rt)
        self.add_edge(parent, rt, weight=weights['RT'])
        self.edge[(self.root, rt)] = 'RT'

    def _replies(self, parent: Tweet, graph_data: dict = {}, level: int = 1):
        """
        Adds a reply node to the graph and its corresponding interactions
        recursively.

        Arguments
        ----
            - parent (Tweet): Tweet (node) of which we want to add the reply.
            - graph_data (dict, optional): Dictionary with the reply
            interactions info. This dict works as a JSON. Defaults to {}.
            - level (int, optional): _description_. Defaults to 1.
        """
        id = int(graph_data['id'])
        tweet = self.db.get_tweets_by_ids(id, ['id', 'author_id', 'text',
                                               'created_at']).iloc[0]
        rp = Reply(tweet)
        self.add_node(rp)
        self.add_edge(parent, rp, weight=weights['RP'])
        self.edge[(parent, rp)] = 'RP'
        self._add_node_interactions(rp, graph_data, False)

    def get_max_branch_nodes(self) -> list:
        """
        Returns the nodes of the max branch of the tree.

        Returns
        -------
            - `List`: nodes of the longest branch of the tree.
        """
        return nx.algorithms.dag.dag_longest_path(self)

    def get_leaf_nodes(self):
        """
        Returns the leaf nodes of the tree, i.e. the nodes that have no
        children.

        Returns
        -------
            - `List`: leaf nodes of the tree.
        """
        sheed_nodes = [node.created_at for node in self.nodes
                       if self.out_degree(node) == 0]

        return sheed_nodes

    def show(self, verbose: bool = False):
        """
        Shows the graph with the different interaction types.

        Arguments
        ----------
            - verbose (bool, optional): Indicates if the program should show
            the information in the nodes of the graph (id, text, author).
        """
        layout = getattr(nx, select_layout())
        attributes = {'color': [], 'size': []}
        for node in self.nodes:
            attributes['color'].append(colors[type(node).__name__])
            attributes['size'].append(sizes[type(node).__name__])

        pos = layout(self, **layouts_attr[layout])
        nx.draw_networkx(self, pos, node_size=attributes['size'],
                         node_color=attributes['color'],
                         node_shape='o', font_size=5,
                         alpha=0.5, with_labels=verbose)

        plt.box(False)
        fig = plt.gcf()
        fig.set_size_inches(32, 18)
        plt.subplots_adjust(left=-0.07, right=1.0, top=1.0, bottom=0.0)

        path = 'images/tree/interactions/'
        name = f'{self.root.id}{layout.__name__.split("_")[0]}'
        show_or_save(plt, path, name)

    def show_emotion(self):
        """
        Shows the graph wtth the different emotions/semtiments of the tweets.
        """
        _type = EmotionType[select_emotion_classifier()]
        layout = getattr(nx, select_layout())
        attributes = {'color': [], 'size': []}
        self.stats = {}
        if _type == EmotionType.BIPOLAR:
            from transformers import pipeline
            classifier = pipeline('sentiment-analysis',
                                  model='nlptown/bert-base-multilingual-' +
                                  'uncased-sentiment')
            self.__bipolar_clasifier(attributes, classifier)
        else:
            self.__multiple_clasifier(attributes)

        pos = layout(self, **layouts_attr[layout])
        nx.draw_networkx(self, pos, node_size=attributes['size'],
                         node_color=attributes['color'],
                         node_shape='o', font_size=5,
                         alpha=0.7)

        plt.box(False)
        fig = plt.gcf()
        fig.set_size_inches(32, 18)
        plt.subplots_adjust(left=-0.07, right=1.0, top=1.0, bottom=0.0)

        path = 'images/tree/tone/'
        name = f'{self.root.id}{str(_type).split(".")[-1]}'
        show_or_save(plt, path, name)

    def __bipolar_clasifier(self, attributes: dict, classifier):
        """
        Classifies the tweets in the graph using the classifier
        passed as argument. Habitually, the classifier is a
        bipolar classifier.

        Arguments
        ----------
            - attributes (dict): Dictionary with the attributes of the nodes
            for this classifier type.
            - classifier: Classifier to use.
        """
        for node in self.nodes:
            results = classifier(node.text)
            stars = int(results[0]['label'].split()[0])
            emotion = list(bipolar_colors.keys())[stars - 1]
            node.emotion = emotion
            self.stats.update({emotion: self.stats.get(emotion, 0) + 1})

            attributes['color'].append(bipolar_colors[emotion])
            attributes['size'].append(800)

    def __multiple_clasifier(self, attributes: dict):
        """
        Classfiies the tweets in the graph using the multiclass classifier.

        Arguments
        ----------
            - attributes (dict): Dictionary with the attributes of the nodes
            for this classifier type.
        """
        for node in self.nodes:
            text_object = NRCLex(node.text)
            emotions = text_object.top_emotions

            max_value = max([emot_val[1] for emot_val in emotions])

            emotions = list(filter(lambda x: x[1] == max_value, emotions)) \
                if max_value != 0 else []
            emotion = sample(emotions, 1)[0] if emotions else ('neutral', 0)
            node.emotion = emotion

            if node.emotion[0] == 'neutral':
                self.stats.update({
                    emotion[0]: self.stats.get(emotion[0], 0) + 1
                })
            else:
                [
                    self.stats.update({
                        emotion[0]: self.stats.get(emotion[0], 0) + emotion[1]
                    }) for emotion in emotions
                ]

            attributes['color'].append(emot_colors[node.emotion[0]])
            attributes['size'].append(300 if node.emotion[0] == 'neutral'
                                      else 1000)

    def get_statistics(self) -> dict:
        """
        Shows some statistics of the conversation graph (start, end,
        maximum depht and number of tweets).

        Returns
        -------
            - `Dict`: Dictionary with some statistics of the graph.
        """
        print('Graph statistics\n' + '-'*35)

        statistics = {}
        start = self.root.created_at
        statistics['start'] = start
        print(f'   - Start: {start}')

        end = max(self.get_leaf_nodes())
        statistics['end'] = end
        print(f'   - End: {end}')

        max_depht = len(self.get_max_branch_nodes())
        statistics['max_depht'] = max_depht
        print(f'   - Maximum Depht: {max_depht}')

        num_nodes = len(self.nodes)
        statistics['num_tweets'] = num_nodes
        print(f'   - Number of Tweets: {num_nodes}')

        return statistics

    def get_emotion_stats(self) -> pd.DataFrame:
        """
        Returns a dataframe with a summary of the emotions in
        the conversation. show_emotion() must be called before.

        Returns
        -------
            - `DataFrame`: Dataframe with the emotions summary.
        """
        _class = self.stats.keys()
        values = np.array(list(self.stats.values()))
        num_tweets = sum(values)
        df = pd.DataFrame({'Class': _class, 'Values': values,
                           'Percentage': values / num_tweets})

        return df
