# CS 175 Winter 2022 - Reddit Trend Analyzer
# Cullen P.P. Moana
# Sushmasri Katakam
# Ethan H. Nguyen
# The following module will be to run the software for the TA

from metrics import *

EXAMPLE_MODELS = ['lexiconanalyzer']
print("Executing main.py...")

# Run Lexicon Analyzer Predictions:
create_predictions(get_lexicon_predictions, write_predictions, "lexiconanalyzer")

# Create Metrics on created Predictions

# Create Weekly Averages for all models
for model in EXAMPLE_MODELS:
    for subreddit in SUBREDDITS:
        write_weekly_avgs(model, subreddit)

# Create Monthly Averages for all models
for model in EXAMPLE_MODELS:
    for subreddit in SUBREDDITS:
        write_monthly_avgs(model, subreddit)

# Create Weekly Totals
for model in EXAMPLE_MODELS:
    for subreddit in SUBREDDITS:
        write_weekly_totals(model, subreddit)

# Create Monthly Totals
for model in EXAMPLE_MODELS:
    for subreddit in SUBREDDITS:
        write_monthly_totals(model, subreddit)

# Print out all statistics calculated
for metric_func in METRIC_FUNCS:
    for subreddit in SUBREDDITS:
        for model in EXAMPLE_MODELS:
            metric_func(model, subreddit)