import pandas as pd
import os

os.chdir("/home/Nickfis/Documents/Projects/HSV_tweets")
all_tweets = pd.read_csv('complete_tweets.csv')

all_tweets.head()

# overall 165965 tweets in the dataset
all_tweets.shape

# translation
from googletrans import Translator
translator = Translator()
# checking how long the translation takes
subset = pd.Series(all_tweets['text'])

# an example to see whether it works.
translator.translate("Sechs Mal Deutscher Meister, drei Mal Pokalsieger, egal welche Liga, HSV!", src='de', dest='en').text

all_tweets.isnull().any()

import copy
from googletrans import Translator


len(all_tweets['text'])


import time
start_time = time.time()
# do it in chunks
i = 0
while i <= 161000:
    # initiliazing the api (by doing it again and again it does not stop)
    translator = Translator()
    translatedList = []
    dateList = []

    for j in all_tweets['text'].iloc[i:(i+5000)]:
        #print(j)
        # REINITIALIZE THE API
        translator = Translator()
        try:
            # translate the 'text' column
            translatedList.append(translator.translate(j, src='de', dest='en').text)

        except Exception as e:
            print(str(e))
            continue

    pd.DataFrame(translatedList).to_csv('translated_tweets' + str(i) + '.csv', index=False)
    i += 5000

print("--- %s seconds ---" % (time.time() - start_time))

######################## push together all the downloaded tweet chunks and adding the dates to them
from os import listdir

filenames = listdir(os.getcwd())

csv_files = [ filename for filename in filenames if filename.endswith('0.csv') ]

# gotta combine the date with the tweets and then concat all the translated tweets together
import re

csv_files = sorted(csv_files, key=lambda x:int(re.findall('(?<=translated_tweets)\d+', x)[0]))
pd.read_csv(csv_files[1])

all_tweets.head()
final_tweets = []
for i in csv_files:
    print(i)
    # reading in the chunks
    translated_tweets = pd.read_csv(i)
    # getting the index
    tweet_index = int(re.findall('(?<=translated_tweets)\d+', i)[0])
    tweet_end = tweet_index + len(translated_tweets)
    # using the index to get the date from the all_tweets_df
    #translated_tweets['date'] = 0
    translated_tweets['date'] = all_tweets['date'].iloc[tweet_index:tweet_end].reset_index(drop=True)
    print(all_tweets['date'].iloc[tweet_index])
    # append the df to the list to create overall frame
    final_tweets.append(translated_tweets)

# taking all dataframes from the list, concatenate them into the final dataframe.
final_df = pd.concat(final_tweets)

# save it to disk.
final_df.reset_index(drop=True).to_csv('all_tweets_en.csv', index=False)
