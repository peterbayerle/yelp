import os
import json
from pandas import DataFrame, read_json
from langdetect import detect

cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('../yelp_dataset/review.json', cur_path)

chunksize = 1000
reader = read_json(new_path, lines=True, chunksize=chunksize)
reviews = []
for chunk in reader:
    for i in range(chunksize):
        reviews.append(chunk.loc[:, "text"][i])
    break
print("reviews read")

#bag of words model
def bag_of_words(rev):
    filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    rev_l = rev.split()
    bag = {}
    for word in rev_l:
        word = word.strip(filters).lower()
        if word in bag.keys():
            bag[word] += 1
        else:
            bag[word] = 1
    return bag

bags = []
for rev in reviews:
    bags.append(bag_of_words(rev))
print("bags computed")

bag_df = DataFrame(bags)
bag_df = bag_df.loc[:, (bag_df.isnull().sum(axis=0) <= chunksize * (1 - 0.05))]
print(len(bag_df.columns))
