import pandas
from pymongo import MongoClient

client = MongoClient("mongodb+srv://root:admin123@jacluster-6afi9.mongodb.net/test?retryWrites=true&w=majority")
db = client.webScience
db2 = client.webScienceTwo
tweets = db.tweets
tweets2 = db2.tweets

cursor = tweets.find()
cursor2 = tweets2.find()

docs = list(cursor)
docs2 = list(cursor2)

df = pandas.DataFrame(columns=[])
df2 = pandas.DataFrame(columns=[])
df_both = pandas.DataFrame(columns=[])

for num, doc in enumerate(docs):
    doc["_id"] = str(doc["_id"])
    doc_id = doc["_id"]
    series_obj = pandas.Series( doc, name=doc_id )
    df.append(series_obj)
    df_both.append(series_obj)

for num, doc in enumerate(docs2):
    doc["_id"] = str(doc["_id"])
    doc_id = doc["_id"]
    series_obj = pandas.Series( doc, name=doc_id )
    df2.append(series_obj)
    df_both.append(series_obj)


json_export = df.to_json("tweets.json")
json_export = df2.to_json("tweets2.json")
json_export = df_both.to_json("tweets_both.json")