# Yelp Review Classifier 👨‍🍳⭐️

Model to classify Yelp reviews as positive (>=3.5 stars) or negative (<3.5 stars) using only their text.

I've converted this project into a [tutorial](https://github.com/peterbayerle/huggingface_notebook) featured in the official 🤗Transformers documentation

## 1. Results
Model performance when trained on 10k reviews and tested on 10k reviews:

|Model|Training Time|Testing Time|Accuracy|
|-|-|-|-|
|Logistic regression|~1 min|~1 min|87.98%|
|DistilBERT|~6.5 hrs|~1.5 hrs|92.52%|

DistilBERT Precision/Recall/F1:

|Class (sentiment)|Precision|Recall|F1|Support|
|-----|---------|------|--|-------|
|Negative|0.9241|0.8510|0.8860|3417|
|Positive|0.9257|0.9637|0.9443|6583|

(more exhaustive performance results for each model are in `./performance`)

## 2. Training details
* 20k random reviews were taken from the [Yelp review dataset](https://www.yelp.com/dataset). Half of these reviews were used for training and the other half for testing.
* The [logistic regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) is used as a baseline for comparison with DistilBERT. Standard normalization techniques including removal of stopwords/punctuaction and lemmatization were used prior to tf-idf vectorization and then training. See `./models/baseline/` for implementation.
* The [🤗Transformers implementation of DistilBERT and the word piece tokenizer](https://huggingface.co/transformers/model_doc/distilbert.html) were used for training the transformer neural net. Reviews were set to a max length of 500. A batch size of 16 and only 1 epoch were used for training (due to limited computational resources 😢). See `./models/tfmr/` for implementation.

## 3. Review visualization
Distribution of ~190,000 restaurants' star ratings
<img src="static/star_distribution.png">
