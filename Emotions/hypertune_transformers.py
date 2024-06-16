import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, Embedding, MultiHeadAttention, GlobalAveragePooling1D
from keras.optimizers import Adam
from transformers import BertTokenizerFast
from emotions_tokens import additional_tokens
import numpy as np
import keras_tuner as kt

embedding_dim = 256  # Embedding dimension
max_sequence_length = 200  # Example sequence length, adjust as needed
tokenizer_vocab_size = 30522  # Example vocabulary size, adjust as needed
num_classes = 28  # Adjust based on your dataset

import pandas as pd
train = pd.read_parquet("go_emotions_train.parquet")
test = pd.read_parquet("go_emotions_test.parquet")
emotions = train.columns[1:29].values
print(emotions)



tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
tokenizer.add_tokens(additional_tokens)

train_sequences = tokenizer(train['text'].tolist(), 
    padding='max_length', truncation=True, 
    max_length=max_sequence_length, return_tensors='np', 
    return_token_type_ids=False, 
    return_attention_mask=False)
train_labels = train[emotions].values
print(len(train_sequences['input_ids']))

X_train = train_sequences['input_ids']
y_train = train_labels

test_sequences = tokenizer(test['text'].tolist(),
    padding='max_length', truncation=True, 
    max_length=max_sequence_length, return_tensors='np', 
    return_token_type_ids=False, 
    return_attention_mask=False)
test_labels = test[emotions].values

X_test = test_sequences['input_ids']
y_test = test_labels

print(len(test_sequences['input_ids']))


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_transformer_model(hp):
    inputs = Input(shape=(max_sequence_length,))
    embedding_layer = Embedding(input_dim=tokenizer_vocab_size, output_dim=embedding_dim, input_length=max_sequence_length)(inputs)
    x = embedding_layer

    # Add a variable number of transformer_encoder blocks
    for i in range(hp.Int('num_transformer_blocks', 1, 4)):
        x = transformer_encoder(
            x,
            head_size=hp.Choice(f'head_size_{i}', values=[128, 256, 512]),
            num_heads=hp.Choice(f'num_heads_{i}', values=[4, 8, 16]),
            ff_dim=hp.Choice(f'ff_dim_{i}', values=[128, 256, 384]),
            dropout=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)
        )
    
    x = GlobalAveragePooling1D()(x)
    x = Dropout(hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    x = Dense(hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu')(x)
    x = Dropout(hp.Float('dropout_4', min_value=0.1, max_value=0.5, step=0.1))(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# HyperModel
strategy = tf.distribute.MirroredStrategy()

# Tuner
tuner = kt.RandomSearch(
    build_transformer_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='transformer_tuning',
    distribution_strategy=strategy
)

tuner.search(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_test, y_test))