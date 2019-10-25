import json

yelp_file = open("yelp_dataset/formatted/business.json")
data = json.load(yelp_file)

for business in data["businesses"][:100]:
    print(business["name"], business["stars"])

yelp_file.close()
