import pandas as pd
import numpy as np
import os

os.chdir("/home/Nickfis/Documents/Projects/HSV_tweets")

np.random.seed(200)
# reading in training set
train = pd.read_csv('trainingset.csv', engine='python', header=None).sample(1000)

train.head()
############################################### data preprocessing ###############################################
# only need the tweet and the sentiment for training
train = train.iloc[:, [0,5]]
colnames = ['sentiment', 'tweet']
train.columns = colnames

# check for the legnth of the tweets
train['length'] = [len(t) for t in train.tweet]
# we have a max of 150, although there are only 140 characters allowed (at that time) on twitter
train['length'].describe()
# for the tweets that have above 140 characters, we can see that there are problems with html encoding.
train[train['length']>140]
# checking for tweet 569987
train.loc[569987].tweet
# we have to clean this up.Furthermore mentions, urls and hashtags will be removed in the cleaning process
# example1 = BeautifulSoup(train.loc[569987].tweet, 'lxml')
# # way better.
# print(example1.get_text())
# len(example1.get_text())

from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
import re

# tweet cleaner
tok = WordPunctTokenizer()
mentions = r'@[A-Za-z0-9]+' # want to get rid of mentions
urls = r'https?://[A-Za-z0-9./]+' # want to delete urls
combined_pat = r'|'.join((mentions, urls))

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml').get_text() # taking care of html encoding
    cleaned = re.sub(combined_pat, '', soup) # delete all mentions and urls
    cleaned = re.sub("[^a-zA-Z]", ' ', cleaned).lower() # get rid of all punctuation and make it lowercase.
    # This adds however a whitespace whenever the regex is matched. But we can't use '' as replacement because otherwise all words will be joined.
    # Therefore we are going to spit the tweet into each word and then join them using only one whitespace to get a cleaned version with exactly one space between each word.
    word_list = tok.tokenize(cleaned)
    return ' '.join(word_list).strip()


testing = train.tweet.iloc[:100]
test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))

# now use the tweet_cleaner to process all the tweets in the train set
# check for the time on the subset to see how long we can expect it to run for all tweets




######## classifier training
# At first we have to train a classifier, prove that it generally works on tweets and then
tweets = pd.read_csv('all_tweets_en.csv')

tweets.columns = ['text', 'date']

del tweets
