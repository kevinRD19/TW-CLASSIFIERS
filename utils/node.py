from utils.utils import clean_replies, clean_text, clean_quote


class Tweet:
    """
    Class that represents a tweet as a node.
    """
    def __init__(self, tweet_info: dict) -> 'Tweet':
        """
        Class constructor.

        Arguments
        ----------
            - tweet_info (`dict`): dictionary that contains the tweet info.
        """
        self.id = int(tweet_info['id'])
        self.author_id = int(tweet_info['author_id'])
        self.text = clean_text(str(tweet_info['text']))
        self.created_at = tweet_info['created_at']
        self.emotion = None

    def __str__(self) -> str:
        """
        Returns a string representation of the tweet.

        Returns
        -------
            - `str`: string that contains the tweet info in a human-readable
            format. Contain the tweet id, user id and the text. If the tweet is
            a number node only contains the text. In emotion nodes, returns an
            empty string.
        """
        return f'Tweet:\n{self.id}\nUsuario: {self.author_id}\n' +\
               f'Texto:\n{self.text[:20]}' if not self.emotion and \
               self.id != self.author_id else self.text \
               if not self.emotion else ''


class Quote(Tweet):
    """
    Class that represents a quote as a node. Have the same behaviour as a
    tweet node.
    """
    def __init__(self, tweet_info: dict) -> Tweet:
        tweet_info['text'] = clean_quote(tweet_info['text'])
        super().__init__(tweet_info)


class Reply(Tweet):
    """
    Class that represents a reply as a node. Have the same behaviour as a
    tweet node.
    """
    def __init__(self, tweet_info: dict) -> Tweet:
        tweet_info['text'] = clean_replies(tweet_info['text'])
        super().__init__(tweet_info)


class Retweet:
    """
    Class that represents a retweet as a node.
    """
    def __init__(self, user_id: int) -> 'Retweet':
        """
        Class constructor.

        Arguments
        ----------
            - user_id (`int`): id of the user that retweeted the tweet.
        """
        self.user_id = user_id

    def __str__(self) -> str:
        """
        Returns a string representation of the retweet.

        Returns
        -------
            `str`: string that contains the user id.
        """
        return f'Usuario\n{self.user_id}'


class Like:
    """
    Class that represents a like as a node.
    """
    def __init__(self, user_id: int) -> 'Like':
        """
        Class constructor.

        Arguments
        ----------
            - user_id (`int`): id of the user that liked a tweet.
        """
        self.user_id = user_id

    def __str__(self) -> str:
        """
        Returns a string representation of the like.

        Returns
        -------
            `str`: string that contains the user id.
        """
        return f'Usuario\n{self.user_id}'
