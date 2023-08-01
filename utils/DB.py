import json
import re
from typing import List
import pandas as pd
from pandas import DataFrame, Series
from sqlalchemy import (Table, create_engine, MetaData,
                        func, select, text)
from sqlalchemy.exc import SQLAlchemyError


class DB:
    """
    Class to handle the connection to the database and the queries to it.
    """
    base = None

    def __init__(self, db_uri: str) -> None:
        """
        Constructor for DB class. It creates a connection to the database
        and set the metadata for the tables.

        Arguments
        ----------
            - db_uri (`str`): URI for the database
        """
        self.db_engine = create_engine(db_uri)
        self.db_metadata = MetaData()
        self.db_conn = self.db_engine.connect()

        self.tweets = None

    def close_connection(self):
        """
        Close the connection to the database
        """
        if self.db_conn:
            self.db_conn.close()

    def get_tweets(self, limit: int = None) -> DataFrame:
        """
        Gets the tweets from the database limited by the limit parameter

        Arguments
        ----------
            - limit (`int`, `optional`): Limit of tweets to get.
            Defaults to None.

        Returns
        -------
            - `DataFrame`: tweets from the database
        """
        tweet = Table('tweet', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        try:
            men_patt = re.compile(r'^mentions_\w+$')
            mentions = list(filter(lambda t: men_patt.match(t),
                            tweet.columns.keys()))
            cols = [col for col in tweet.c if col.name not in mentions]
            query = select(*cols).where(tweet.c.lang == 'es')\
                .limit(limit) if limit \
                else select(*cols).where(tweet.c.lang == 'es')
            tweets = self.db_conn.execute(query).fetchall()

            return pd.DataFrame(tweets)

        except SQLAlchemyError as err:
            print(err)
            return None

    def get_tweets_metrics(self) -> DataFrame:
        """
        Gets the tweet metrics from the database

        Returns
        -------
            - `Dataframe`: tweet metrics (retweets, replies,
            quotes and likes)
        """

        tw_metrics = Table('tweet_metrics', self.db_metadata,
                           autoload_replace=True,
                           autoload_with=self.db_engine)
        query_ = select(tw_metrics)
        metrics = self.db_conn.execute(query_).fetchall()

        df_metrics = pd.DataFrame(metrics)
        df_metrics['interaction_count'] = df_metrics['retweet_count'] + \
            df_metrics['reply_count'] + df_metrics['quote_count'] + \
            df_metrics['like_count']

        return df_metrics

    def get_count_tweets(self) -> DataFrame:
        """
        Gets the count of tweets per author, sorted by tweet count

        Returns
        -------
            - `Dataframe`: count of tweets per author
        """
        if self.tweets is None:
            self.get_tweets()
        try:
            if not self.db_count_tw:
                pass
        except AttributeError:
            self.df_count_tw = self.tweets.groupby('author_id')['author_id'].\
                count().reset_index(name='count')
            self.df_count_tw.sort_values(by=['count'], ascending=False,
                                         inplace=True)
            self.df_count_tw.reset_index(inplace=True, drop=True)
        return self.df_count_tw

    def get_most_hashtags(self) -> DataFrame:
        """
        Gets the most used hashtags sorted by the count of uses

        Returns
        -------
            - `DataFrame`: most used hashtags
        """
        hashtag = Table('hashtag', self.db_metadata, autoload_replace=True,
                        autoload_with=self.db_engine)
        hashtagt_tw = Table('hashtagt_tweet', self.db_metadata,
                            autoload_replace=True,
                            autoload_with=self.db_engine)
        join_ = hashtag.join(hashtagt_tw)
        query_ = select(hashtag.c.hashtag, func.count().label('num_uses'))\
            .select_from(join_).group_by(hashtag.c.hashtag)\
            .order_by(text('num_uses DESC'))

        count_hasht = self.db_conn.execute(query_).fetchall()
        self.df_count_hasht = pd.DataFrame(count_hasht).reset_index(drop=True)

        return self.df_count_hasht

    def get_most_campaign(self) -> DataFrame:
        """
        Gets the campaigns with the most tweets sorted by tweet count

        Returns
        -------
            - `DataFrame`: most used campaigns
        """
        tweet = Table('tweet', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        query = select(tweet.c.campaign, func.count().label('num_uses'))\
            .group_by(tweet.c.campaign).order_by(text('num_uses DESC'))

        count_campaign = self.db_conn.execute(query).fetchall()
        self.df_count_cmp = pd.DataFrame(count_campaign).reset_index(drop=True)

        return self.df_count_cmp

    def get_most_categories(self) -> DataFrame:
        """
        Gets the most used categories sorted by the use count

        Returns
        -------
            - `Dataframe`: most used categories
        """
        ann_tw = Table('annotation_tweet', self.db_metadata,
                       autoload_replace=True, autoload_with=self.db_engine)
        domain = Table('domain', self.db_metadata,
                       autoload_replace=True, autoload_with=self.db_engine)
        join_ = ann_tw.join(domain)
        query_ = select(domain.c.category, func.count().label('num_uses'))\
            .select_from(join_).group_by(domain.c.category)\
            .order_by(text('num_uses DESC'))

        count_category = self.db_conn.execute(query_).fetchall()
        self.df_count_cat = pd.DataFrame(count_category).reset_index(drop=True)

        return self.df_count_cat

    def get_most_person(self) -> DataFrame:
        """
        Gets the different persons mentioned in the tweets and the number of
        times they are mentioned, sorted by the number of mentions.

        Returns
        -------
            - `DataFrame`: most mentioned persons
        """
        ann_tw = Table('annotation_tweet', self.db_metadata,
                       autoload_replace=True, autoload_with=self.db_engine)
        domain = Table('domain', self.db_metadata,
                       autoload_replace=True, autoload_with=self.db_engine)
        join_ = ann_tw.join(domain)
        query_ = select(domain.c.name, func.count().label('num_uses'))\
            .select_from(join_).where(domain.c.category.like('Person'))\
            .group_by(domain.c.name).order_by(text('num_uses DESC'))

        count_name = self.db_conn.execute(query_).fetchall()
        self.df_count_person = pd.DataFrame(count_name).reset_index(drop=True)

        return self.df_count_person

    def get_top_users(self) -> DataFrame:
        """
        Gets the authors (id and username) with its respective number of posts
        (tweets, retweets, replies, quotes) ordered by the number of posts.

        Returns
        -------
            - `DataFrame`: top of the active users
        """
        tweet = Table('tweet', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        user = Table('user', self.db_metadata, autoload_replace=True,
                     autoload_with=self.db_engine)
        posts_count = select(tweet.c.author_id.label('user_id'),
                             func.count().label('num_posts'))\
            .group_by(tweet.c.author_id).alias('posts_count')
        join_ = user.join(posts_count)
        query_ = select(text('posts_count.user_id, username, num_posts'))\
            .select_from(join_)\
            .order_by(text('num_posts DESC'))
        num_posts = self.db_conn.execute(query_).fetchall()
        self.df_num_posts = pd.DataFrame(num_posts).reset_index(drop=True)

        return self.df_num_posts

    def get_tweet_by_user(self, user_id: int | List) -> DataFrame:
        """
        Gets the tweets of the given user(s).

        Arguments:
        ----------
            - user_id (`int` | `List`): id or list of ids of the users to
            get the tweets from.

        Returns:
        --------
            - `DataFrame`: tweets of the given users
        """
        tweet = Table('tweet', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        if isinstance(user_id, int):
            user_id = [user_id]

        query = select(tweet.c.author_id, tweet.c.created_at)\
            .where(tweet.c.author_id.in_(user_id))
        tweets = self.db_conn.execute(query).fetchall()

        return pd.DataFrame(tweets).reset_index(drop=True)

    def get_tweets_by_ids(self, tweet_id: int | List, columns: List = []) -> \
            DataFrame:
        """
        Gets the tweet information of the given tweet(s).

        Arguments:
        ----------
            - tweet_id (`int` | `List`): id or list of ids of the tweets to
            get the information from.

        Returns:
        --------
            - `DataFrame`: tweets information that match the given ids
        """
        tweet = Table('tweet', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)

        if isinstance(tweet_id, int):
            tweet_id = [tweet_id]
        columns = set(columns) & set(tweet.columns.keys()) \
            if columns else tweet.columns.keys()
        columns = [col for col in tweet.c if col.name in columns]

        query = select(*columns).where(tweet.c.id.in_(tweet_id))
        tweets = self.db_conn.execute(query).fetchall()

        return pd.DataFrame(tweets).reset_index(drop=True)

    def get_tweets_by_hashtag(self, hashtag: str | List) -> DataFrame:
        """
        Gets the tweets that contain the given hashtag(s).

        Arguments:
        ----------
            - hashtag (`str` | `List`): hashtag or list of hashtags to get the
            tweets from.

        Returns:
        --------
            - `DataFrame`: tweets that contain at least one of the given
        """
        tweet = Table('tweet', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        hs_tw = Table('hashtagt_tweet', self.db_metadata,
                      autoload_replace=True,
                      autoload_with=self.db_engine)
        hs = Table('hashtag', self.db_metadata,
                   autoload_replace=True,
                   autoload_with=self.db_engine)

        if isinstance(hashtag, str):
            hashtag = [hashtag]
        join_ = hs_tw.join(hs).join(tweet, tweet.c.id == hs_tw.c.tweet_id)
        query = select(hs.c.hashtag, tweet.c.created_at)\
            .select_from(join_).where(hs.c.hashtag.in_(hashtag))
        tweets = self.db_conn.execute(query).fetchall()

        return pd.DataFrame(tweets).reset_index(drop=True)

    def get_tweets_by_mention(self, names: str | List) -> DataFrame:
        """
        Gets the tweets that mention the given name(s).

        Arguments:
        ----------
            - names (`str` | `List`): name(s) that must be mentioned in the
            tweets.

        Returns:
        --------
            - DataFrame: tweets that mention at least one of the given names.
        """
        tweet = Table('tweet', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        domain = Table('domain', self.db_metadata, autoload_replace=True,
                       autoload_with=self.db_engine)
        ann_tw = Table('annotation_tweet', self.db_metadata,
                       autoload_replace=True,
                       autoload_with=self.db_engine)
        if isinstance(names, str):
            names = [names]
        join_ = ann_tw.join(tweet).join(domain,
                                        ann_tw.c.domain_id == domain.c.id)
        query = select(domain.c.name, tweet.c.created_at)\
            .select_from(join_).where(domain.c.name.in_(names))

        tweets = self.db_conn.execute(query).fetchall()

        return pd.DataFrame(tweets).reset_index(drop=True)

    def get_users(self) -> DataFrame:
        """
        Gets all user id from the database.

        Returns:
        -------
            - `DataFrame`: dataframe with the user id.
        """
        user = Table('user', self.db_metadata, autoload_replace=True,
                     autoload_with=self.db_engine)
        users = self.db_conn.execute(select(user.c.id)).fetchall()

        return pd.DataFrame(users,
                            columns=['author_id']).reset_index(drop=True)

    def get_retweets(self) -> DataFrame:
        """
        Gets the retweets from the database

        Returns:
        --------
            - `DataFrame`: retweets information
        """
        rt = Table('retweet', self.db_metadata, autoload_replace=True,
                   autoload_with=self.db_engine)
        retweets = self.db_conn.execute(select(rt)).fetchall()

        return pd.DataFrame(retweets).reset_index(drop=True)

    def get_replies(self) -> DataFrame:
        """
        Gets the replies from the database

        Returns:
        --------
            - `DataFrame`: replies information
        """
        reply = Table('reply', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        replies = self.db_conn.execute(select(reply)).fetchall()

        return pd.DataFrame(replies).reset_index(drop=True)

    def get_quotes(self) -> DataFrame:
        """
        Gets the quotes from the database

        Returns:
        --------
            - `DataFrame`: quotes information
        """
        quoted = Table('quoted', self.db_metadata, autoload_replace=True,
                       autoload_with=self.db_engine)
        quotes = self.db_conn.execute(select(quoted)).fetchall()

        return pd.DataFrame(quotes).reset_index(drop=True)

    def get_posts_by_type(self, user_id: int | List) -> DataFrame:
        """
        Gets the number of tweets, retweets, replies and quotes done by
        the given users.

        Arguments:
        ----------
            - user_id (`int` | `List`): id or list of ids of the users to get
            the posts from.

        Returns:
        --------
            - `DataFrame`: dataframe with the information of the posts
        """
        tweet = Table('tweet', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        retweet = Table('retweet', self.db_metadata, autoload_replace=True,
                        autoload_with=self.db_engine)
        reply = Table('reply', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        quoted = Table('quoted', self.db_metadata, autoload_replace=True,
                       autoload_with=self.db_engine)
        if isinstance(user_id, int):
            user_id = [user_id]
        rt = select(tweet.c.author_id.distinct().label('user_id'),
                    func.count().label('num_rt')) \
            .select_from(tweet.join(retweet))\
            .where(tweet.c.author_id.in_(user_id))\
            .group_by(tweet.c.author_id).alias('rt')
        qt = select(tweet.c.author_id.distinct().label('user_id'),
                    func.count().label('num_qt'))\
            .select_from(tweet.join(quoted))\
            .where(tweet.c.author_id.in_(user_id))\
            .group_by(tweet.c.author_id).alias('qt')
        rp = select(tweet.c.author_id.distinct().label('user_id'),
                    func.count().label('num_rp')) \
            .select_from(tweet.join(reply)) \
            .group_by(tweet.c.author_id).alias('rp')
        query = select(text('rt.user_id, num_rt, num_qt, num_rp'))\
            .select_from(rt.join(rp, rt.c.user_id == rp.c.user_id,
                                 isouter=True)
                           .join(qt, rt.c.user_id == qt.c.user_id,
                                 isouter=True))
        posts = self.db_conn.execute(query).fetchall()

        return pd.DataFrame(posts).reset_index(drop=True)

    def get_interactions_list(self, ids: int | List) -> DataFrame:
        """
        Gets the list of interactions of the given tweet(s).

        Arguments
        ----------
            - ids (`int` | `List`): id or list of ids of the tweets to get the
            interactions from.

        Returns
        -------
            - DataFrame: information of the interactions of the given tweets
        """
        rt = self._get_retweet_dates_post(ids)
        rp = self._get_reply_dates_post(ids)
        tweets = pd.concat([rt, rp], ignore_index=True)

        return tweets

    def _get_retweet_dates_post(self, ids: int | List) -> DataFrame:
        """
        Gets the dates of the retweets of the given tweet(s).

        Arguments
        ----------
            - id (`int` | `List`): id of the tweet(s) to get the
            retweet dates.

        Returns:
            - `DataFrame`: retweet dates of the given tweets.
        """
        if isinstance(ids, int):
            ids = [ids]
        tweet = Table('tweet', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        retweet = Table('retweet', self.db_metadata, autoload_replace=True,
                        autoload_with=self.db_engine)
        _join = retweet.join(tweet, tweet.c.tweet_id == retweet.c.retweeted)
        rt_list = select(retweet.c.tweet_id.label('id'),
                         tweet.c.id.label('tweet_id'))\
            .select_from(_join).where(tweet.c.id.in_(ids)).alias('rt_list')
        query = select(rt_list.c.tweet_id, tweet.c.created_at)\
            .select_from(rt_list.join(tweet, rt_list.c.id == tweet.c.id))
        tweets = self.db_conn.execute(query).fetchall()

        return pd.DataFrame(tweets).reset_index(drop=True)

    def _get_reply_dates_post(self, ids: int | List) -> DataFrame:
        """
        Gets the dates of the replies of the given tweet(s).

        Arguments
        ----------
            - id (`int` | `List`): id of the tweet(s) to get the reply dates.

        Returns
        -------
            - `DataFrame`: reply dates of the given tweets.
        """
        if isinstance(ids, int):
            ids = [ids]
        tw = Table('tweet', self.db_metadata, autoload_replace=True,
                   autoload_with=self.db_engine)
        rp = Table('reply', self.db_metadata, autoload_replace=True,
                   autoload_with=self.db_engine)
        rp_list = select(rp.c.tweet_id.label('id'), tw.c.id.label('tweet_id'))\
            .select_from(rp.join(tw, tw.c.tweet_id == rp.c.reply_to))\
            .where(tw.c.id.in_(ids)).alias('rp_list')
        query = select(rp_list.c.tweet_id, tw.c.created_at)\
            .select_from(rp_list.join(tw, rp_list.c.id == tw.c.id))
        tweets = self.db_conn.execute(query).fetchall()

        return pd.DataFrame(tweets).reset_index(drop=True)

    def get_tweet_dates_details(self) -> DataFrame:
        """
        Gets the date, author id and tweet id of the tweets.

        Returns:
        --------
            - `DataFrame`: tweet information
        """
        tweet = Table('tweet', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        query = select(tweet.c.id, tweet.c.author_id, tweet.c.created_at)

        tweets = self.db_conn.execute(query).fetchall()

        return pd.DataFrame(tweets).reset_index(drop=True)

    def get_tweets_to_conexion(self) -> DataFrame:
        """
        Gets the tweets that are part of a conversation (that have root tweet)

        Returns
        -------
            - `DataFrame`: tweets that are part of a conversation
        """
        tweet = Table('tweet', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        query = select(tweet.c.id, tweet.c.tweet_id, tweet.c.author_id,
                       tweet.c.conversation_id, tweet.c.lang)\
            .where(tweet.c.conversation_id.in_(
                select(tweet.c.tweet_id.distinct())
            ))
        tweets = self.db_conn.execute(query).fetchall()

        return pd.DataFrame(tweets).reset_index(drop=True)

    def get_root_tweets(self, df: DataFrame) -> DataFrame:
        """
        Gets the independent root tweets of the conversations,
        that is, the tweets that are not retweets, replies or quotes.

        Arguments
        ----------
            - df (`DataFrame`): tweets to study

        Returns
        -------
            - `DataFrame`: lndependent root tweets
        """
        rt = Table('retweet', self.db_metadata, autoload_replace=True,
                   autoload_with=self.db_engine)
        rp = Table('reply', self.db_metadata, autoload_replace=True,
                   autoload_with=self.db_engine)
        qt = Table('quoted', self.db_metadata, autoload_replace=True,
                   autoload_with=self.db_engine)
        rem_rt_query = select(rt.c.tweet_id)
        tweets_to_remove = [t for t, in self.db_conn.execute(rem_rt_query)
                                                    .fetchall()]
        df = df[~df['id'].isin(tweets_to_remove)]

        rem_rp_query = select(rp.c.tweet_id)
        tweets_to_remove = [t for t, in self.db_conn.execute(rem_rp_query)
                                                    .fetchall()]
        df = df[~df['id'].isin(tweets_to_remove)]

        rem_qt_query = select(qt.c.tweet_id)
        tweets_to_remove = [t for t, in self.db_conn.execute(rem_qt_query)
                                                    .fetchall()]
        df = df[~df['id'].isin(tweets_to_remove)]

        return df

    def get_sources_count(self) -> DataFrame:
        """
        Gets the sources of the tweets and the number of tweets that have that
        source.

        Returns
        -------
            - `DataFrame`: sources and number of tweets that have each source
        """
        tweet = Table('tweet', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        query = select(tweet.c.source, func.count().label('num_tweets'))\
            .group_by(tweet.c.source).order_by(text('num_tweets DESC'))
        sources = self.db_conn.execute(query).fetchall()

        return pd.DataFrame(sources).reset_index(drop=True)

    def get_tree(self, df: DataFrame, tweet: Series, isroot: bool = False,
                 replies: bool = True) -> tuple[dict, DataFrame]:
        """
        Get the tree of the conversation of the given tweets.

        Arguments
        ----------
            - df (`Dataframe`): all the tweets to study.
            - tweet (`Series`): tweet to get the tree from.
            - isroot (`bool`, `optional`): indicates if the given tweet
            is the root of the conversation. Defaults to False.
            - replies (`bool`, `optional`): indicates if the replies of the
            tweet must be included in the tree. Defaults to True.

        Returns
        -------
            - `dict`: tree of the conversation of the given tweet.
            - `DataFrame`: tweets to study after delete tweets in the tree.
        """
        tree = json.loads('{}')
        if isroot:
            tree['conversation_id'] = int(tweet['conversation_id'])
            tree['root'] = int(tweet['id'])
        else:
            tree['id'] = int(tweet['id'])
        rt, df = self._get_rt(df, tweet)
        tree.update({'retweets': rt})
        qt, df = self._get_qt(df, tweet)
        tree.update({'quotes': qt})
        like = self._get_like(tweet)
        tree.update({'likes': like})
        if replies:
            rp, df = self._get_rp(df, tweet)
            tree.update({'replies': rp})

        return tree, df.reset_index(drop=True)

    def _get_rt(self, df: DataFrame, tweet: Series) -> tuple[list, DataFrame]:
        """
        Gets the users that retweeted the given tweet.

        Arguments
        ----------
            - df (`Dataframe`): all the tweets to study.
            - tweet (`Series`): tweet data to get the retweets from.

        Returns
        -------
            - `list`: id of the users that retweeted the given tweet.
            - `DataFrame`: tweets to study after delete the retweets.
        """
        rt = Table('retweet', self.db_metadata, autoload_replace=True,
                   autoload_with=self.db_engine)
        rt_query = select(rt.c.tweet_id)\
            .where(rt.c.retweeted == tweet['tweet_id'])
        retweets = [t for t, in self.db_conn.execute(rt_query).fetchall()]
        retweets = df[df['id'].isin(retweets)][['id', 'author_id']]
        df = df[~df['id'].isin(retweets['id'].tolist())]

        return retweets['author_id'].tolist(), df

    def _get_qt(self, df: DataFrame, tweet: Series) -> tuple[list, DataFrame]:
        """
        Gets the quotes of the given tweet.

        Arguments
        ----------
            - df (`Dataframe`): all the tweets to study.
            - tweet (`Series`): tweet data to get the quotes from.

        Returns
        -------
            - `list`: id of the quotes of the given tweet.
            - `DataFrame`: tweets to study after delete the quotes.
        """
        qt_list = []
        qt = Table('quoted', self.db_metadata, autoload_replace=True,
                   autoload_with=self.db_engine)
        qt_query = select(qt.c.tweet_id)\
            .where(qt.c.quoted == tweet['tweet_id'])
        quotes = [t for t, in self.db_conn.execute(qt_query).fetchall()]
        quotes = df[df['id'].isin(quotes)]
        df = df[~df['id'].isin(quotes['id'].tolist())]
        for i, quote in quotes.iterrows():
            tree, df = self.get_tree(df, quote)
            qt_list.append(tree)

        return qt_list, df

    def _get_rp(self, df: DataFrame, tweet: Series) -> tuple[list, DataFrame]:
        """
        Gets the replies of the given tweet.

        Arguments
        ----------
            - df (`DataFrame`): dataframe that contains all the tweets
            to study.
            - tweet (`Series`): tweet data to get the replies from.

        Returns
        -------
            - `list`: id of the replies of the given tweet.
            - `DataFrame`: tweets to study after delete the replies.
        """
        replies_list = []
        poss_rps = df.loc[df['conversation_id'] == tweet['conversation_id']]
        df = df[~df['id'].isin(poss_rps['id'].tolist())]
        replies = poss_rps.loc[poss_rps['id'] != tweet['id']]
        for _, reply in replies.iterrows():
            tree, df = self.get_tree(df, reply, replies=False)
            replies_list.append(tree)

        return replies_list, df

    def _get_like(self, tweet: Series) -> list:
        """
        Gets the users that liked the given tweet.

        Arguments
        ----------
            - tweet (`Series`): tweet data to get the likes from.

        Returns
        -------
            - `list`: user ids that liked the given tweet.
        """
        like = Table('like_tweet', self.db_metadata, autoload_replace=True,
                     autoload_with=self.db_engine)
        like_query = select(like.c.user_id)\
            .where(like.c.tweet_id == tweet['id'])
        likes = [t for t, in self.db_conn.execute(like_query).fetchall()]
        return likes

    def get_tweet_text(self) -> DataFrame:
        """
        Gets the text of the tweets

        Returns
        -------
            - `DataFrame`: tweets data including the text
        """
        tweet = Table('tweet', self.db_metadata, autoload_replace=True,
                      autoload_with=self.db_engine)
        query = select(tweet.c.id, tweet.c.text, tweet.c.lang,
                       tweet.c.author_id)
        tweets = self.db_conn.execute(query).fetchall()
        tweets = pd.DataFrame(tweets).reset_index(drop=True)
        tweets['text'] = tweets['text'].apply(lambda x: re.sub(r'\n+', ' ', x))
        tweets['text'] = tweets['text'].apply(lambda x: re.sub(r'\s+', ' ', x))

        return tweets

    def hashtag_usage_statics(self) -> DataFrame:
        """
        Gets the number of tweets that use each hashtag.

        Returns
        -------
            - `DataFrame`: usage information of the hashtags.
        """
        tw = Table('tweet', self.db_metadata, autoload_replace=True,
                   autoload_with=self.db_engine)
        ht_tw = Table('hashtagt_tweet', self.db_metadata,
                      autoload_replace=True, autoload_with=self.db_engine)

        join_ = tw.join(ht_tw)
        query_ = select(tw.c.author_id, func.count().label('used_hashtags'))\
            .select_from(join_).group_by(tw.c.author_id, tw.c.id)\
            .order_by(text('used_hashtags DESC'))
        statics = self.db_conn.execute(query_).fetchall()

        return pd.DataFrame(statics).reset_index(drop=True)

    def get_users_info(self, users: List[int],
                       columns: List[str] = []) -> DataFrame:
        """
        Gets the requested information of the given users.

        Arguments
        ----------
            - users (`List[int]`): ids of the users to get the information
            - columns (`List[str]`, `optional`): columns to get from the
            user table. Defaults to [].

        Returns
        -------
            - `DataFrame`: required information of the given users
        """
        user = Table('user', self.db_metadata, autoload_replace=True,
                     autoload_with=self.db_engine)
        columns = set(columns) & set(user.columns.keys()) \
            if columns else user.columns.keys()
        columns = [col for col in user.c if col.name in columns]

        users = self.db_conn.execute(select(*columns).
                                     where(user.c.id.in_(users))).fetchall()

        return pd.DataFrame(users).reset_index(drop=True)
