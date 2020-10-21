#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from scipy import stats
from operator import itemgetter
import json
import re
import collections
import preprocessor as p


# In[ ]:


def read_raw_tweets(tweets_data_path):
    tweets_data = []  
    tweets_file = open('tweets_solar_20-02-17.txt', "r")  
    for line in tweets_file:  
        try:  
            tweet = json.loads(line)  
            tweets_data.append(tweet)
        except:  
            continue
    tweets_data = [x for x in tweets_data if not isinstance(x, int)]
    num = len(tweets_data)
    print("\n[INFO] Number of read tweets: {:,}".format(num))
    return tweets_data

def raw_tw_to_pd(tweets_data):
    tweets = pd.DataFrame()
    tweets['user_id'] = list(map(lambda tweet: tweet['user']['id'], tweets_data))
    tweets['user'] = list(map(lambda tweet: tweet['user']['screen_name'], tweets_data))
    tweets['name'] = list(map(lambda tweet: tweet['user']['name'], tweets_data))
    tweets['description'] = list(map(lambda tweet: tweet['user']['description'], tweets_data))
    tweets['verified'] = list(map(lambda tweet: tweet['user']['verified'], tweets_data))
    tweets['protected'] = list(map(lambda tweet: tweet['user']['protected'], tweets_data))
    tweets['location'] = list(map(lambda tweet: tweet['user']['location'], tweets_data))
    tweets['lang'] = list(map(lambda tweet: tweet['user']['lang'], tweets_data))
    tweets['followers'] = list(map(lambda tweet: tweet['user']['followers_count'], tweets_data))
    tweets['following'] = list(map(lambda tweet: tweet['user']['friends_count'], tweets_data))
    tweets['favourites'] = list(map(lambda tweet: tweet['user']['favourites_count'], tweets_data))
    tweets['lists'] = list(map(lambda tweet: tweet['user']['listed_count'], tweets_data))
    tweets['tweets_cnt'] = list(map(lambda tweet: tweet['user']['statuses_count'], tweets_data))
    tweets['acc_creation'] = list(map(lambda tweet: tweet['user']['created_at'], tweets_data))
    tweets['default_profile'] = list(map(lambda tweet: tweet['user']['default_profile'], tweets_data))
    tweets['default_prof_image'] = list(map(lambda tweet: tweet['user']['default_profile_image'], tweets_data))
    tweets['image'] = list(map(lambda tweet: tweet['user']['profile_image_url'], tweets_data))
    tweets['notifications'] = list(map(lambda tweet: tweet['user']['notifications'], tweets_data))
    
    tweets['text'] = list(map(lambda tweet: p.clean(tweet['text']) if 'extended_tweet' not in tweet else p.clean(tweet['extended_tweet']['full_text']), tweets_data))
    tweets['quoted_text'] = list(map(lambda tweet: p.clean(tweet['text']) if 'quoted_status' not in tweet else p.clean(tweet['quoted_status']['text']), tweets_data))
    tweets['tweet_id'] = list(map(lambda tweet: tweet['id'], tweets_data))
    tweets['created_at'] = list(map(lambda tweet: tweet['created_at'], tweets_data))
    tweets['length'] = list(map(lambda tweet: len(tweet['text']) if'extended_tweet' not in tweet else len(tweet['extended_tweet']['full_text']) , tweets_data))
    tweets['retweet_cnt'] = list(map(lambda tweet: tweet['retweet_count'], tweets_data))
    tweets['favorite_cnt'] = list(map(lambda tweet: tweet['favorite_count'], tweets_data))
    tweets['reply_to_twt_id'] = list(map(lambda tweet: tweet['in_reply_to_status_id'], tweets_data))
    tweets['reply_to_user'] = list(map(lambda tweet: tweet['in_reply_to_screen_name'], tweets_data))
    tweets['reply_to_user_id'] = list(map(lambda tweet: tweet['in_reply_to_user_id'], tweets_data))
    tweets['coordinates'] = list(map(lambda tweet: tweet['coordinates'], tweets_data))
    
    tweets['device'] = list(map(reckondevice, tweets_data))
    tweets['RT'] = list(map(is_RT, tweets_data))
    tweets['Reply'] = list(map(is_Reply_to, tweets_data))   

    num = tweets.shape[0]
    print("\nNumber of Tweets: {:,}".format(num))
    print("From: {} to {}\n".format(tweets.created_at[0], tweets.created_at[num-1]))
    return tweets

