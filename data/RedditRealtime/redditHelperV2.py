# CS 175 Winter 2022 - Reddit Trend Analyzer
# Cullen P.P. Moana
# Sushmasri Katakam
# Ethan H. Nguyen

import requests
import json


for subreddit in ['UCI', 'Russia', 'Ukraine']:
  r = requests.get(f'https://api.pushshift.io/reddit/submission/search/?after=1626897110&before=1627761110&sort_type=score&sort=desc&subreddit={subreddit}&pretty=true')

  print(subreddit + '--------------------------------------')
  for i in range(20):
    print(r.json()['data'][i]['title'])