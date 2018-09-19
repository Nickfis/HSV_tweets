import pandas as pd
import numpy as np
import os
import matplotlib as plt

os.chdir("/home/Nickfis/Documents/Projects/HSV_tweets")

np.random.seed(1887)
# reading in training set
train = pd.read_csv('trainingset.csv', engine='python', header=None, encoding='ISO-8859–1')

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
#train.loc[852964].tweet
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
train['tweet'] = train['tweet'].apply(lambda x: normalizer(x))
print("--- %s seconds ---" % (time.time() - start_time))

#train.to_csv('cleaned_up_train.csv', index=False)

train = pd.read_csv('cleaned_up_train.csv')
############################################### data preprocessing: tokenizing, stemming, lemmatization ###############################################
# removing stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')

train['tweet'] = train['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv.fit(train.tweet)

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
frequency.sort_values(by='negative', ascending=False).head(10)
frequency.sort_values(by='positive', ascending=False).head(10)

# Another metric is the frequency a words occurs in the class. This is defined as
pos_freq = train[train['sentiment']==4].shape[0]

# pos_porcentage will now hold the amount of tweets a word can be found in divided by the overall number of positive tweets
# pos_frequenc of that word / positive_frequency_tweets
frequency['pos_percentage'] = frequency['positive'] / pos_freq

# now we want to combie both the positive rate and the appearance in positive tweets of that word into one metric
# since pos_percentage is a way smaller value in general we will have to do some scaling, so that it does not get overpowered by the pos_rate value
# We will therefore treat both columns as samples as an approximation of their probability distrbution. Scaling them between 0-1, giving each word basically its
# percentile inside the CDF as its value.
from scipy.stats import norm
def normcdf(x):
    return norm.cdf(x, x.mean(), x.std())

# creating the cdf for the pos_rate column
frequency['pos_rate_normcdf'] = normcdf(frequency['pos_rate'])
# for the pos_percentage column
frequency['pos_percentage_normcdf'] = normcdf(frequency['pos_percentage'])
# taking the mean of both columns to create the new metric
frequency['pos_normcdf'] = frequency[['pos_rate_normcdf', 'pos_percentage_normcdf']].mean(axis=1)

frequency.head()

# order according to newly created metric
frequency.sort_values(by='pos_normcdf', ascending=False)

# creating the same metric now for negative tweets

# create negative rate first
frequency['neg_rate'] = frequency['negative'] / frequency['overall_frequency']
# percentage in its class
neg_freq = train[train['sentiment']==0].shape[0]
frequency['neg_percentage'] = frequency['negative'] / neg_freq

# doing column calculations for creation of negative sentiment metrics
frequency['neg_rate_normcdf'] = normcdf(frequency['neg_rate'])
# for the pos_percentage column
frequency['neg_percentage_normcdf'] = normcdf(frequency['neg_percentage'])
# taking the mean of both columns to create the new metric
frequency['neg_normcdf'] = frequency[['neg_rate_normcdf', 'neg_percentage_normcdf']].mean(axis=1)
# sorting by that to check strongest negative words
frequency.sort_values(by='neg_normcdf', ascending=False)
# seems to work well from first impression
import seaborn as sns
from pylab import *
plt.figure(figsize=(8,6))
ax = sns.regplot(x="neg_normcdf", y="pos_normcdf",fit_reg=False, scatter_kws={'alpha':0.5},data=frequency)
plt.ylabel('Positive Rate and Frequency CDF')
plt.xlabel('Negative Rate and Frequency CDF')
plt.title('neg_normcdf_hmean vs pos_normcdf_hmean')
# getting rid of length column
train = train[['tweet', 'sentiment']]


# split into train, validation and test

# since the dataset has overall 1.6 million observations, a validation and testset of 1% still gives us about 15.000 observations, which is enough to test the trained classifer.
# On the other hand this gives us over 1.5 million observations to train the classifer on.

train.head()
train.loc[train['sentiment']==4, 'sentiment'] = 1

df = train
del train


x = df['tweet']
y = df['sentiment']
from sklearn.cross_validation import train_test_split

seed = 1887

X_train, X_validation, y_train, y_validation = train_test_split(x,y, test_size=0.02, random_state=seed)

X_validation, X_test, y_validation, y_test = train_test_split(X_validation, y_validation, test_size=0.5, random_state=seed)

X_train.shape
X_validation.shape
X_test.shape

## Feature extraction

# create a (limited) count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cvec = CountVectorizer()
logreg = LogisticRegression()
n_features = 20000

sen
data_pipeline = Pipeline([
                        ('vectorizer', cvec),
                        ('classifier', logreg)
                        ])
data_pipeline
def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):
    start_time = time()
    sentiment_fit = pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_test)
    end_time = time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    return end_time, accuracy


timing, accuracy_result = accuracy_summary(data_pipeline, X_train, y_train, X_test, y_test)

timing
accuracy_result
5+3









# pos_tagging: first part of speech tagging for stemming of the used words in the tweets
import nltk
# tokenization
def tokenizer(tweet):
    theTokens = re.findall(r'\b\w[\w-]*\b', tweet)
    return theTokens

# example what we have to do now in order to find stem for each word
text = tokenizer(train['tweet'].iloc[3])
nltk.pos_tag(text2)






######## classifier training
# At first we have to train a classifier, prove that it generally works on tweets and then
tweets = pd.read_csv('all_tweets_en.csv')

tweets.columns = ['text', 'date']

del tweets
