import json
import inquirer
import os
import sys
from pprint import pprint
from itertools import chain

c_dir = os.path.dirname(os.path.abspath(__file__))
p_dir = os.path.dirname(c_dir)
sys.path.append(p_dir)

from utils.utils import CONFIG # noqa

# Gets the label types and subtypes from the config file
labels = CONFIG['protest_types']
# Gets the tweets to label
with open('data/labeled_protests.json', 'r') as f:
    tweets = json.load(f)

# Counts the number of labeled tweets for each subtype
counts = {}
for tweet in tweets:
    if 'subtype' in tweet:
        counts.update({
            tweet['subtype']: counts.get(tweet['subtype'], 0) + 1
        })
used_labels = len(counts.keys())
subtypes = [
    labels[keys] for keys in labels.keys()
]
subtypes = list(chain.from_iterable(subtypes))
counts = {
    key: counts[key] if key in counts else 0
    for key in subtypes
}

# Shows the number of labeled tweets for each subtype
pprint(counts)
# Summary of labeled tweets
print('\nSummary of labeled tweets\n' + '-' * 38)
print('   - Number of tweets:', len(tweets))
print('   - Total labeled tweets:', sum(counts.values()))
print(f'   - Subtypes without labeled tweets: {len(subtypes)-used_labels}\n')

# Asks for the type and subtype of each tweet if it is not labeled
for tweet in tweets:
    if 'type' not in tweet:
        types = ['CANCEL', 'NEXT'] + list(labels.keys())
        _type = inquirer.prompt([inquirer.List('type', message=tweet['text'],
                                               choices=types)])
        # If the user wants to exit the program
        if _type['type'] == 'CANCEL':
            if inquirer.confirm('Do you want to exit?',
                                default=False):
                break
            else:
                continue
        # If the user wants to skip the tweet
        elif _type['type'] == 'NEXT':
            continue

        subtypes = ['CANCEL'] + list(labels[_type['type']])
        subtype = inquirer.prompt(
            [inquirer.List('subtype', message=tweet['text'],
                           choices=subtypes)])
        if subtype['subtype'] == 'CANCEL':
            if inquirer.confirm('Do you want to exit?',
                                default=False):
                break
            else:
                continue
        tweet['type'] = _type['type']
        tweet['subtype'] = subtype['subtype']
        print(tweet)

# Saves the labeled tweets
with open('data/labeled_protests.json', 'w') as f:
    json.dump(tweets, f, indent=4, ensure_ascii=False)
