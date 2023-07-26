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

labels = CONFIG['protest_types']

with open('data/labeled_protests.json', 'r') as f:
    tweets = json.load(f)

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

pprint(counts)
print('Number of tweets:', len(tweets))
print('Total Labeled:', sum(counts.values()))
print('Subtypes Without Label:', len(subtypes) - used_labels)

for tweet in tweets:
    if 'type' not in tweet:
        types = ['CANCEL', 'NEXT'] + list(labels.keys())
        _type = inquirer.prompt([inquirer.List('type', message=tweet['text'],
                                               choices=types)])

        if _type['type'] == 'CANCEL':
            if inquirer.confirm('Do you want to exit?',
                                default=False):
                break
            else:
                continue
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

with open('data/labeled_protests.json', 'w') as f:
    json.dump(tweets, f, indent=4, ensure_ascii=False)
