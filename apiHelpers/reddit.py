# CS 175 Winter 2022 - Reddit Trend Analyzer
# Cullen P.P. Moana
# Sushmasri Katakam
# Ethan H. Nguyen

import praw

reddit = praw.Reddit(
  client_id = "NcnZ1Hlq-30m8DoVBgSF3A",
  client_secret = "Fw3KZnzH6fmlaz8O0ro719Z004Kr_w",
  user_agent = "python:com.sentimentanalyzer:v1.0.0",
)

for submission in reddit.subreddit("UCI").hot(limit=10):
    print(submission.title)

