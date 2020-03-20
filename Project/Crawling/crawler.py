import tweepy
import mongo

#https://chatbotslife.com/crawl-twitter-data-using-30-lines-of-python-code-e3fece99450e
#code built on this

consumer_key = "dLCwlq9fnnMsC3MubxNf7Xbyh"
consumer_secret = "hReDjRF6in1rNJgxVQfvAqBVLgnDHRkpki5rz2W9Dc3ZxDxM8C"
access_token = "1230486587621990404-fA8Iq7eQnKR7u1Rp5HJV2tZudV6UHs"
access_secret = "V1rrNebZbZ3FBeHWUnG11CyIYB5b2ww61x0qbgMcHOIS0"
tweetsPerQry = 100
maxTweets = 100000
hashtag = "#glasgow"
duplicates= 0

authentication = tweepy.OAuthHandler(consumer_key, consumer_secret)
authentication.set_access_token(access_token, access_secret)
api = tweepy.API(authentication, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
maxId = -1
tweetCount = 0
while tweetCount < maxTweets:
    if(maxId <= 0):
        newTweets = api.search(q=hashtag, count=tweetsPerQry, result_type="recent")
    else:
        newTweets = api.search(q=hashtag, count=tweetsPerQry, max_id=str(maxId - 1), result_type="recent")

    if not newTweets:
        print("Finished")
        break
	
    for tweet in newTweets:
        mongo.addTweet(tweet, duplicates, api)

    tweetCount += len(newTweets) 
    maxId = newTweets[-1].id