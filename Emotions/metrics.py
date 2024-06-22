import tensorflow as tf
import time 
import keras

DEBUG=True
tfprint = print


f1 = keras.metrics.F1Score("micro")
hinge = keras.metrics.CategoricalHinge(name="hinge")
rac = keras.metrics.R2Score(name="R2")

falsePositives = keras.metrics.FalsePositives(name="false_positives")
falseNegatives = keras.metrics.FalseNegatives(name="false_negatives")
truePositives = keras.metrics.TruePositives(name="true_positives")
trueNegatives = keras.metrics.TrueNegatives(name="true_negatives")

precision = keras.metrics.Precision(name="precision")
recall = keras.metrics.Recall(name="recall")

precisionAtRecal = keras.metrics.PrecisionAtRecall(recall=0.5, name="precision_over_recall")

auc = keras.metrics.AUC(multi_label=True, num_labels=28, name="auc")
roc = keras.metrics.AUC(multi_label=True, num_labels=28, curve="ROC", name="roc")

binary_crossentropy = keras.losses.BinaryCrossentropy(from_logits=False)


all_metrics = [f1, hinge, rac, falsePositives, falseNegatives, truePositives, trueNegatives, precision, recall, precisionAtRecal, auc, roc]




def cosine_similarity_loss(y_true, y_pred):
    if DEBUG:
        tfprint("Type of y_true:", tf.shape(y_true))
        tfprint("Type of y_pred:", tf.shape(y_pred))
        tfprint("y_true:", y_true)
        tfprint("y_pred before normalization:", y_pred)

    # Add a small epsilon to embeddings before normalization to avoid division by zero
    y_pred = y_pred + tf.keras.backend.epsilon()

    # L2 normalize embeddings
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    y_true = tf.math.l2_normalize(y_true, axis=1)

    if DEBUG:
        tfprint("Normalized y_pred:", y_pred)
        tfprint("Normalized y_true:", y_true)

    # Compute cosine similarity
    cosine_similarity = tf.reduce_sum(y_pred * y_true, axis=-1)

    if DEBUG:
        tfprint("Cosine Similarity:", cosine_similarity)

    # Compute the loss
    loss = -tf.reduce_mean(cosine_similarity)

    if DEBUG:
        tfprint("Loss:", loss)

    return loss


class ExamplesPerSecondMetric(tf.keras.metrics.Metric):
    def __init__(self, name='examples_per_second', **kwargs):
        super(ExamplesPerSecondMetric, self).__init__(name=name, **kwargs)
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')
        self.start_time = self.add_weight(name='start_time', initializer='zeros')
        self.epoch_time = self.add_weight(name='epoch_time', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.shape(y_true)[0]
        self.total_samples.assign_add(tf.cast(batch_size, tf.float32))
        if self.start_time == 0:
            self.start_time.assign(time.time())
    
    def result(self):
        self.epoch_time.assign(time.time() - self.start_time)
        examples_per_second = self.total_samples / self.epoch_time
        return examples_per_second
    
    def reset_state(self):
        self.total_samples.assign(0)
        self.start_time.assign(0)
        self.epoch_time.assign(0)