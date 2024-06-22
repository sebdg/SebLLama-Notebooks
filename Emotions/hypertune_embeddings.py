import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, Embedding, MultiHeadAttention, GlobalAveragePooling1D
from keras.optimizers import Adam
import numpy as np
import keras_tuner as kt
from load_data import load_tokenizer, load_data
from metrics import cosine_similarity_loss, all_metrics

max_sequence_length = 200  # Example sequence length, adjust as needed
tokenizer = load_tokenizer()
X_train, y_train, X_test, y_test = load_data(tokenizer, max_sequence_length)

def build_embeddings_model(hp):
    inputs = Input(shape=(max_sequence_length,), dtype=tf.int32)
    x = Embedding(tokenizer.vocab_size, hp.Int('embedding_dim', 128, 512, 32), input_length=max_sequence_length)(inputs)
    x = GlobalAveragePooling1D()(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(hp.Int('dense_units', 128, 512, 32), activation='relu')(x)
    x = Dropout(hp.Float('dropout', 0.1, 0.5, 0.1))(x)
    outputs = Dense(28, activation='sigmoid')(x)
    model = Model(inputs, outputs, name='embeddings_model')
    model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5], default=1e-3)),
                  loss="binary_crossentropy",
                  metrics=[ "accuracy"])
    return model

# HyperModel
strategy = tf.distribute.MirroredStrategy()

# Tuner
tuner = kt.Hyperband(
    build_embeddings_model,
    objective="val_accuracy",
    max_epochs=100,
    executions_per_trial=1,
    directory='runs',
    project_name='embbedings_tuning',
    distribution_strategy=strategy
)

tuner.search(X_train, y_train, batch_size=1024, validation_data=(X_test, y_test))