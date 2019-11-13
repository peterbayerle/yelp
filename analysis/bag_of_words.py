import os
import json
from pandas import DataFrame, read_json

cur_path = os.path.dirname(__file__)
new_path = os.path.relpath('../yelp_dataset/review.json', cur_path)

chunksize = 100000
reader = read_json(new_path, lines=True, chunksize=chunksize)
count = 0
for chunk in reader:
    for i in range(chunksize):
        print(chunk.loc[:, "business_id"][i])
    break

filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
rev_l = rev.split()

bag = {}
for word in rev_l:
    word = word.strip(filters).lower()
    if word in bag.keys():
        bag[word] += 1
    else:
        bag[word] = 1
#
# print(bag)

# reviews = DataFrame(data["reviews"]).fillna(0)
