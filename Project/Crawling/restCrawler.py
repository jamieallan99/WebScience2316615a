import tweepy
import pymongo

client = pymongo.MongoClient("mongodb+srv://root:admin123@jacluster-6afi9.mongodb.net/test?retryWrites=true&w=majority")
db = client.webScienceTwo

#https://chatbotslife.com/crawl-twitter-data-using-30-lines-of-python-code-e3fece99450e
#code built on this

consumer_key = "dLCwlq9fnnMsC3MubxNf7Xbyh"
consumer_secret = "hReDjRF6in1rNJgxVQfvAqBVLgnDHRkpki5rz2W9Dc3ZxDxM8C"
access_token = "1230486587621990404-fA8Iq7eQnKR7u1Rp5HJV2tZudV6UHs"
access_secret = "V1rrNebZbZ3FBeHWUnG11CyIYB5b2ww61x0qbgMcHOIS0"
tweetsPerQry = 10
maxTweets = 100000
hashtag = "#CoronaUK"
duplicates= 0

authentication = tweepy.AppAuthHandler(consumer_key, consumer_secret)
#authentication.set_access_token(access_token, access_secret)
api = tweepy.API(authentication, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

tweets= db.tweets

query= {'retweet_id': {'$regex':'([0-9])*'}}

documents = tweets.find()

for tweet in documents:
    try:
        print(tweet['retweet_id'])
    except:
        pass
    