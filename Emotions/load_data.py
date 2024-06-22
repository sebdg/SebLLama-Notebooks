import pandas as pd
from transformers import BertTokenizerFast
from emotions_tokens import additional_tokens
import numpy as np

def load_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    tokenizer.add_tokens(additional_tokens)
    return tokenizer


def load_data(tokenizer, max_sequence_length=200):
    train = pd.read_parquet("go_emotions_train.parquet")
    test = pd.read_parquet("go_emotions_test.parquet")
    emotions = train.columns[1:29].values
    train_sequences = tokenizer(train['text'].tolist(), 
        padding='max_length', truncation=True, 
        max_length=max_sequence_length, return_tensors='np', 
        return_token_type_ids=False, 
        return_attention_mask=False)
    train_labels = train[emotions].values

    test_sequences = tokenizer(test['text'].tolist(),
        padding='max_length', truncation=True, 
        max_length=max_sequence_length, return_tensors='np', 
        return_token_type_ids=False, 
        return_attention_mask=False)
    test_labels = test[emotions].values

    X_train = train_sequences['input_ids']
    y_train = train_labels.astype(np.float32)
    X_test = test_sequences['input_ids']
    y_test = test_labels.astype(np.float32)
    return X_train, y_train, X_test, y_test
