import pandas as pd
import numpy as np
import os
import matplotlib as plt

os.chdir("/home/Nickfis/Documents/Projects/HSV_tweets")

np.random.seed(200)
# reading in training set
train = pd.read_csv('trainingset.csv', engine='python', header=None, encoding='ISO-8859–1').sample(10000)

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
train.loc[972155].tweet
# we have to clean this up.Furthermore mentions, urls and hashtags will be removed in the cleaning process
# example1 = BeautifulSoup(train.loc[569987].tweet, 'lxml')
# # way better.
# print(example1.get_text())
# len(example1.get_text())

from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
import re

# clean tweets and normalize them
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



def normalizer(text):
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


# testing = train.tweet.iloc[:100]
# test_result = []
# for t in testing:
#     test_result.append(tweet_cleaner(t))
#
# test_result

# now use the tweet_cleaner to process all the tweets in the train set
# check for the time on the subset to see how long we can expect it to run for all tweets
import time
start_time = time.time()
# cleaning
train['tweet'] = train['tweet'].apply(lambda x: tweet_cleaner(x))
print("--- %s seconds ---" % (time.time() - start_time))

############################################### data preprocessing: tokenizing, stemming, lemmatization ###############################################
# removing stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')

train['tweet'] = train['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv.fit(train.tweet)

neg_cv

# looking at the use of each of the words found split up between tweets labelled as positive and negative
# transofrm function to turn the sparse matrix cv into a dense one
neg_cv = cv.transform(train[train['sentiment'] == 0].tweet) # 0 encoded as negative sentiment
pos_cv = cv.transform(train[train['sentiment'] == 4].tweet) # 4 encoded as positive
neg_tf = np.sum(neg_cv,axis=0) # summing over all observed values
pos_tf = np.sum(pos_cv,axis=0)
neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))

# create the frequency matrix using all words in the set (by transposing making them the rows) and show frequencies in columns
frequency = pd.DataFrame([neg,pos],columns=cv.get_feature_names()).transpose()

frequency.columns = ['negative', 'positive']

frequency['overall_frequency'] = frequency['negative'] + frequency['positive']
# sort by overall frequency to see the most used words in the dataset
frequency.sort_values(by='overall_frequency', ascending=False)
# positive rate: how often is a word used in a positive tweet to overall mentions? Use it as feature in later model
frequency['pos_rate'] = frequency['positive'] / frequency['overall_frequency']
# what's the distribution of our words?
frequency['overall_frequency'].describe()
frequency['overall_frequency'].hist(bins=100, range=(0,50))

# look at top 10 words most used in positive and negative tweets
frequency.head()
frequency.sort_values(by='negative', ascending=False).head(10)
frequency.sort_values(by='positive', ascending=False).head(10)
# pos_tagging: first part of speech tagging for stemming of the used words in the tweets
import nltk
# tokenization
def tokenizer(tweet):
    theTokens = re.findall(r'\b\w[\w-]*\b', tweet)
    return theTokens

# example what we have to do now in order to find stem for each word
text = tokenizer(train['tweet'].iloc[3])
nltk.pos_tag(text2)

## adding more and more features to the analysis for the classifier first. # analysis part 3 has to be finished.


5+3
train.head()



######## classifier training
# At first we have to train a classifier, prove that it generally works on tweets and then
tweets = pd.read_csv('all_tweets_en.csv')

tweets.columns = ['text', 'date']

del tweets
