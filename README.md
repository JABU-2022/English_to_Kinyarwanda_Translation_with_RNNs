# English to Kinyarwanda Translation using RNNs

This project aims to build a model that translates text from English to Kinyarwanda using Recurrent Neural Networks (RNNs).

## Dataset
The dataset used for this project was sourced from publicly available resources:
- [Dataset 1](https://huggingface.co/datasets/DigitalUmuganda/kinyarwanda-english-machine-translation-dataset/resolve/main/kinyarwanda-english-corpus.tsv)
- [Dataset 2](https://huggingface.co/datasets/DigitalUmuganda/kinyarwanda-english-machine-translation-dataset/resolve/main/kinyarwanda-english-corpus2.tsv)
- [Dataset 3](https://huggingface.co/datasets/DigitalUmuganda/kinyarwanda-english-machine-translation-dataset/resolve/main/kinyarwanda-english-corpus3.tsv)

## Preprocessing
- Lowercasing
- Removing Non-Alphanumeric Characters
- Adding Special Tokens (<start>, <end>)
- Tokenization
- Padding

## Model Architecture
The model uses a Bidirectional GRU for the encoder and a GRU for the decoder, with embedding layers for converting integer sequences to dense vectors.

### Encoder

from tensorflow.keras.layers import Embedding, GRU, Bidirectional
from tensorflow.keras.models import Model
import tensorflow as tf

class Encoder(Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = Bidirectional(GRU(self.enc_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform'))

    def call(self, x, hidden):
        x = self.embedding(x)
        output, forward_h, backward_h = self.gru(x, initial_state=hidden)
        state = tf.concat([forward_h, backward_h], axis=-1)
        return output, state

    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_sz, self.enc_units)) for _ in range(2)]

### Decoder

class Decoder(Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.gru = GRU(self.dec_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.fc = Dense(vocab_size)

    def call(self, x, hidden, enc_output):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        output = self.fc(output)
        return output, state

## Training Process and Hyperparameters

### Training Steps

The training involves the following steps

- Batch Size: 64
- Epochs: 20
- Optimizer: Adam
- Learning Rate: 0.001
- Loss Function: Sparse categorical cross-entropy

### Evaluation

The model is evaluated using BLEU scores.

### Results

BLEU Score: 0.7

### Insights and Potential Improvements

- The model performs well on short sentences but struggles with longer and more complex sentences.
- Adding more data and using a more complex model (e.g., Transformer) might improve performance.

### How to Use

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- Pandas
- NLTK

### Instructions

- Clone the repository.
- Install the required packages.
- Run the Jupyter notebook to train the model and evaluate the results.

## Acknowledgements

Special thanks to the Digital Umuganda initiative for providing the dataset and the developers of TensorFlow for their excellent framework.



