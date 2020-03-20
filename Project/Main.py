import tweepy

auth = tweepy.OAuthHandler("dLCwlq9fnnMsC3MubxNf7Xbyh", "hReDjRF6in1rNJgxVQfvAqBVLgnDHRkpki5rz2W9Dc3ZxDxM8C")
auth.set_access_token("1230486587621990404-fA8Iq7eQnKR7u1Rp5HJV2tZudV6UHs", "V1rrNebZbZ3FBeHWUnG11CyIYB5b2ww61x0qbgMcHOIS0")

api = tweepy.API(auth)

try:
    api.verify_credentials()
    print("Authentication OK")
except:
    print("Error during authentication")

"""
for tweet in tweepy.Cursor(api.search, q='tweepy').items(10):
    print(tweet.text)
"""
for status in tweepy.Cursor(api.user_timeline).items(10):
    # process status here
    print(status)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(tweet.text)