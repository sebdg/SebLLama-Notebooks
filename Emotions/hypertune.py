import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input, BatchNormalization, LSTM, Bidirectional
from keras.optimizers import Adam
import keras_tuner as kt
from transformers import BertTokenizerFast
from emotions_tokens import additional_tokens
import numpy as np

embedding_dim = 256  # Embedding dimension
max_sequence_length = 256  # Example sequence length, adjust as needed
tokenizer_vocab_size = 10000  # Example vocabulary size, adjust as needed
num_classes = 28  # Adjust based on your dataset

def build_model(hp):
    model = Sequential([
        Input(shape=(max_sequence_length,)),
        Embedding(input_dim=tokenizer_vocab_size, output_dim=hp.Int('embedding_dim', min_value=32, max_value=256, step=32), input_length=max_sequence_length),
        Conv1D(filters=hp.Int('conv1_filters', min_value=32, max_value=256, step=32), kernel_size=hp.Choice('conv1_kernel', values=[3, 5, 7]), activation='relu'),
        BatchNormalization(),
        Dropout(hp.Float('conv1_dropout', min_value=0.2, max_value=0.5, step=0.1)),
        Conv1D(filters=hp.Int('conv2_filters', min_value=32, max_value=256, step=32), kernel_size=hp.Choice('conv2_kernel', values=[3, 5, 7]), activation='relu'),
        BatchNormalization(),
        Dropout(hp.Float('conv2_dropout', min_value=0.2, max_value=0.5, step=0.1)),
        Bidirectional(LSTM(units=hp.Int('lstm_units', min_value=32, max_value=256, step=32), return_sequences=True)),
        BatchNormalization(),
        Dropout(hp.Float('lstm_dropout', min_value=0.2, max_value=0.5, step=0.1)),
        Bidirectional(LSTM(units=hp.Int('lstm2_units', min_value=32, max_value=256, step=32))),
        BatchNormalization(),
        Dropout(hp.Float('lstm2_dropout', min_value=0.2, max_value=0.5, step=0.1)),
        Dense(units=hp.Int('dense_units', min_value=32, max_value=256, step=32), activation='relu'),
        BatchNormalization(),
        Dropout(hp.Float('dense_dropout', min_value=0.2, max_value=0.5, step=0.1)),
        Dense(num_classes, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='emotion_detection_tuning'
)
import pandas as pd

train = pd.read_parquet("go_emotions_train.parquet")
test = pd.read_parquet("go_emotions_test.parquet")


emotions = train.columns[1:].values

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
tokenizer.add_tokens(additional_tokens)

max_sequence_length = 200

train_sequences = tokenizer(train['text'].tolist(), 
    padding='max_length', truncation=True, 
    max_length=max_sequence_length, return_tensors='np', 
    return_token_type_ids=False, 
    return_attention_mask=False)
train_labels = train[emotions].values
print(len(train_sequences['input_ids']))

X_train = train_sequences['input_ids']
y_train = train_labels.astype(np.float32)

test_sequences = tokenizer(test['text'].tolist(),
    padding='max_length', truncation=True, 
    max_length=max_sequence_length, return_tensors='np', 
    return_token_type_ids=False, 
    return_attention_mask=False)
test_labels = test[emotions].values

X_test = test_sequences['input_ids']
y_test = test_labels.astype(np.float32)



print(len(test_sequences['input_ids']))

tuner.search(X_train, y_train, epochs=10, batch_size=2048, validation_data=(X_test, y_test))