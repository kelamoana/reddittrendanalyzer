import praw
from requests_toolbelt import user_agent

reddit = praw.Reddit(
  client_id = "NcnZ1Hlq-30m8DoVBgSF3A",
  client_secret = "Fw3KZnzH6fmlaz8O0ro719Z004Kr_w",
  user_agent = "python:com.sentimentanalyzer:v1.0.0",
)

print(reddit.read_only)

for submission in reddit.subreddit("UCI").hot(limit=10):
    print(submission.title)
