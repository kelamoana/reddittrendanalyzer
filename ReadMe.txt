In order to run this code for the Software Example:

1) Install Python
2) Navigate to src using cd
3) Run using Python 3.8

In the terminal:
py -3.8 main.py

--------------------------------------------------------------------

Libraries used:

• numpy
• pandas
• nltk
• nltk.
• gensim
• gensim.downloader
• vaderSentiment
• sklearn.model_selection
• sklearn.linear_model
• matplotlib.pyplot
• keras.preprocessing.text
• keras.preprocessing.sequence


# List the publicly available code(s) you used in your project. Please provide the URL for the
# code and mention if you modified the code or not. If you modified the code, please mention
# the number of lines your team modified or added.

Publicly available codes used:
• RNN Intro (https://towardsdatascience.com/a-beginners-guide-on-sentiment-analysis-with-rnn-9e100627c02e) Used to understand Keras API. Not Used in our code.
• RNN Base code (https://towardsdatascience.com/machine-learning-word-embedding-sentiment-classification-using-keras-b83c28087456). Modified/added approximately 20 lines of code
• Reddit API (https://www.reddit.com/dev/api/) Used to create scripts to pull real-life data. Used in 10-15 lines of code.
• Vader Sentiment API (https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664) Used to create code to implement VADER model. Used 5-8 lines of code.
• RNN API (https://www.tensorflow.org/guide/keras/rnn) Used for reference in about 15 lines of code.
• Logistic Regression API (https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8) Referenced to create code. Used in about 20 lines of code overall. 
• MatPlotLib API (https://matplotlib.org/) Used to generate graphs from metrics. Referenced in 10-15 lines of code.

# List the code(s) written entirely by your team. Please roughly mention how
# many lines of code is present in each and provide a brief description (for each) of what the
# code does.

Scripts/functions written by our team:

• logisticRegression.py - About 95% custom - Creates and Trains Logistic Regression Classifier using BOW and TFIDF approach.
• lexiconanalzyer.py - About 15% custom - Creates a Lexicon Analyzer Model capable of determining pos/neg sentiment on a statement.
• main.py - 100% custom - Executes a miniature script that emulates the workflow of our overall project.
• metrics.py - About 95% custom - Implements all learners and uses them to create predictions on real-life data, and then applies metrics to those predictions to analyze for trends.
• recurringNeuralNetworks.py - About 30% custom - Creates and Trains a Recurrent Neural Network capable of making predictions on sentiment for the data.
• helpers.py - About 98% custom - Contains a multitude of helper functions used to process data and create usable datastructures with information that we are using to train.

--------------------------------------------------------------------

IF THE GRADER IS UNABLE TO RUN THE CODE:

We have included PNGs of the outputs of each of our models 
and the metrics associated with them. They are stored in the
graphs folder, and organized by subreddit. The name of each individual
model and metric is attached to each graph, and will provide the statistics 
on our sentiment analysis.

--------------------------------------------------------------------
SIDE NOTE:

In order to use the venv you may have to run a specific command
if your IDE does not automatically use it.

For windows:
C:\> env\Scripts\activate.bat

For mac:
$ source venv/Scripts/activate

~~~~ It is important to use the venv since the libraries will be the 
ones that are compatible with the code, as well as the ones that will
make it work bug-free!!!~~~~



