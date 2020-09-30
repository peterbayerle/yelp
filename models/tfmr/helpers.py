import tensorflow as tf

def construct_feature_list(x, tkzr, max_len):
    feature_list = []
    for text in x:
        inputs = tkzr(
            text,
            max_length=max_len,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            padding='max_length'
        )
        input_ids, attention_masks, token_type_ids = inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']
        feature_list.append((input_ids, attention_masks, token_type_ids))
    return feature_list

def to_tfdataset(feature_list, y=None):
    if y == None:
        y =[-1] * len(feature_list)

    def gen():
        for idx, data in enumerate(feature_list):
            yield ({'input_ids': data[0],
                     'attention_mask': data[1],
                     'token_type_ids': data[2]},
                    y[idx])

    tfdataset = tf.data.Dataset.from_generator(
        gen,
        ({'input_ids': tf.int32,
          'attention_mask': tf.int32,
          'token_type_ids': tf.int32},
         tf.int64),
        ({'input_ids': tf.TensorShape([None]),
          'attention_mask': tf.TensorShape([None]),
          'token_type_ids': tf.TensorShape([None])},
         tf.TensorShape([]))
    )

    return tfdataset
