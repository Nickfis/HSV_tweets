# HSV_tweets

Sentiment analysis in order to extract mood among the fans of the Bundesliga club throughout the historical season which ended with its first relegation in club history.

This was a project done for a text-mining course during my Masters' degree. However, trying to make the sentiment analysis work with German tweets and a German testset
was incredibly hard and did not lead to satisfactory results. Therefore I decided to give it another go by translating the tweets first and then using the (compared to
German resources) extensive English resources for NLP classification.

For downloading the tweets I used a program written by Jefferson-Henrique, that can also find old tweets, which is not possible through the ordinary twitter api.
His repo can be found here: https://github.com/Jefferson-Henrique/GetOldTweets-python 
The preprocessing.py file starts with the dataset obtained by using his program, since this is where the data sciency work starts. 
