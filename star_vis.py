import json
from matplotlib.pyplot import show, subplots
import os

path = os.path.join('data', 'business.json')
yelp_file = open(path)
data = json.load(yelp_file)

rating_freq = {}
for business in data['businesses']:
    stars = business['stars']
    if stars in rating_freq.keys():
        rating_freq[stars] += 1
    else:
        rating_freq[stars] = 1

star_ticks = list(rating_freq.keys())
star_ticks.sort()

fig, ax = subplots()
rects = ax.bar(rating_freq.keys(), rating_freq.values(), width=0.3)
ax.set_xticks(star_ticks)
ax.set_xlabel('Star rating')
ax.set_ylim(top=40_000)
ax.set_title('Yelp star rating distribution')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            str(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords='offset points',
            ha='center', va='bottom'
        )

autolabel(rects)

show()

yelp_file.close()
