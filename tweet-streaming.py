#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tweepy
import json
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from pymongo import MongoClient


# In[2]:


api_key = '......'
api_secret = '......'
access_token = '......'
access_secret = '......'

auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)


# In[3]:


auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

class MyStreamListener(tweepy.StreamListener):
    def __init__(self):
        self.output_file = open('solar_tweets_20-10-18-00.json', 'w')
        
    def on_data(self, data):
        if not data.endswith('\n'): 
            data += '\n'
        
        self.output_file.write(data)
        
    def on_error(self, status):
        print('ERROR', status)
        return False
    
    def on_status(self, status):
        print(status.text)


myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener, tweet_mode='extended')

myStream.filter(track=['solar energy', 'solar panel', 'solar panels', 'solar PV', 'solar rebates', 'solar photovoltaic', 'solar battery', 'solar thermal', 'solar power', 'solar tax', 'solar subsidies', 'solar-powered', 'rooftop solar', 'community solar', 'solar generation'])

# Exclude tweets if they include... 
if 'BTS' in text.lower() or 'mamamoo' in text.lower() or 'eclipse' in text.lower():
    return False  


# In[ ]:




