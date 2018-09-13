import pandas as pd
import numpy as np
import os

os.chdir("/home/Nickfis/Documents/Projects/HSV_tweets")

np.random.seed(200)
# reading in training set
train = pd.read_csv('trainingset.csv', engine='python', header=None, encoding='ISO-8859–1').sample(1000)

train.head()
############################################### data preprocessing: Cleaning up ###############################################
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
train.loc[852964].tweet
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
mentions = r'@[A-Za-z0-9_]+' # want to get rid of mentions
urls = r'https?://[^ ]+' # want to delete urls
combined_pat = r'|'.join((mentions, urls))
www_pat = r'www.[^ ]+' # also delete URLs that start with wwww.
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')



def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml').get_text() # taking care of html encoding
    cleaned = re.sub(combined_pat, '', soup) # delete all mentions and urls
    cleaned = re.sub(www_pat,'', cleaned).lower() # delete www. urls and turn it into lower case
    # using the negations dictionary to take switch all negations into their written out term
    # otherwise when we get rid of punctuation, can't will turn into can t (Which might be equally interpreted as "can" "t")
    negations_clean = neg_pattern.sub(lambda x: negations_dic[x.group()], cleaned)
    cleaned = re.sub("[^a-zA-Z]", ' ', negations_clean) # get rid of all punctuation
    # This adds however a whitespace whenever the regex is matched. But we can't use '' as replacement because otherwise all words will be joined.
    # Therefore we are going to spit the tweet into each word and then join them using only one whitespace to get a cleaned version with exactly one space between each word.
    word_list = tok.tokenize(cleaned)
    return ' '.join(word_list).strip()


testing = train.tweet.iloc[:100]
test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))

test_result

# now use the tweet_cleaner to process all the tweets in the train set
# check for the time on the subset to see how long we can expect it to run for all tweets
import time
start_time = time.time()
# cleaning
train['tweet'] = train['tweet'].apply(lambda x: tweet_cleaner(x))
print("--- %s seconds ---" % (time.time() - start_time))

############################################### data preprocessing: lemmatization,  ###############################################

# tokenization
def tokenizer(tweet):
    theTokens = re.findall(r'\b\w[\w-]*\b', tweet)
    return theTokens





######## classifier training
# At first we have to train a classifier, prove that it generally works on tweets and then
tweets = pd.read_csv('all_tweets_en.csv')

tweets.columns = ['text', 'date']

del tweets
