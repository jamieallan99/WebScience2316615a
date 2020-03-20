import pymongo

client = pymongo.MongoClient("mongodb+srv://root:admin123@jacluster-6afi9.mongodb.net/test?retryWrites=true&w=majority")
db = client.webScienceTwo
duplicates= 0
#tweet will be a dictionary with the following attributes; 

def find_retweeters(id, api):
    retweeters= api.retweeters(id)
    return retweeters

def addTweet(tweet, duplicates, api):
    username= tweet.user.name
    id= tweet.id
    date= tweet.created_at
    text= tweet.text
    source= tweet.source
    
    try:
        coordinates= tweet.coordinates.coordinates
    except:
        coordinates= None

    try:
        quoted_user= tweet.quoted_status.user.name
    except:
        quoted_user= None

    try:
        retweeted_user= tweet.retweeted_status.user.name
        #retweet_users= find_retweeters(id, api)
        retweet_id= tweet.retweeted_status.id
        retweet_users= None
    except:
        retweet_id= None
        retweeted_user= None
        retweet_users= None

    place= tweet.place
    hashtags= tweet.entities['hashtags']
    mongo_tweet= {
        '_id':id,
        'username':username,
        'quoted_user':quoted_user,
        'retweeted_user':retweeted_user,
        'retweet_id':retweet_id,
        'date':date,
        'text':text,
        'source':source,
        'coordinates':coordinates,
        'place':place,
        'hashtags':hashtags,
        'retweet_users':retweet_users
    }
    try:
        result= db.tweets.insert_one(mongo_tweet)
    except:
        duplicates += 1