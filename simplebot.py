# from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
import time

from tensorflow.python.keras.models import load_model


checkpoint_dir = './training_checkpoints'
path_to_file = tf.keras.utils.get_file('mueller.txt', 'http://www.viralml.com/static/code/mueller.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))
idx2char = np.array(vocab)
char2idx = {u:i for i, u in enumerate(vocab)}


def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def generate_text(model, start_string):
  num_generate = 200
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  temperature = 1.0
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      predictions = tf.squeeze(predictions, 0)
      predictions = predictions / temperature
      predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
      input_eval = tf.expand_dims([predicted_id], 0)
      text_generated.append(idx2char[predicted_id])
  return (start_string + ''.join(text_generated))


def load_pretrained_model():
    load_model("my_model.h5")
    model.compile(optimizer = tf.train.AdamOptimizer(), loss=loss)
    return model


if __name__ == "__main__":
    model = load_pretrained_model()
    start_string = u"Potential Section 1030 Violation By "
    print(generate_text(model, start_string))

