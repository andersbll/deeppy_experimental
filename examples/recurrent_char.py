#!/usr/bin/env python

import os
os.environ['CUDARRAY_BACKEND'] = 'numpy'
import cudarray as ca
import numpy as np
import pickle
import joblib
import deeppy as dp
from deeppy.graph.recurrent_graph import (GatedRecurrent, RecurrentGraph,
                                          OneHot, Reshape)

memory = joblib.Memory(cachedir='./cache',)
memory.clear()

def string_to_array(s, vocabulary):
    s = np.array(list(s))
    s_array = np.zeros_like(s, dtype=dp.int_)
    for idx, char in enumerate(vocabulary):
        s_array[s == char] = idx
    return s_array


def array_to_string(s_array, vocabulary):
    return vocabulary[s_array].tostring()


def one_hot_decode(one_hot):
    return np.argmax(one_hot, axis=1)


@memory.cache
def dataset(filename, lowercase=False):
    with open(filename, 'r') as f:
        text = f.read()
    if lowercase:
        text = text.lower()
    vocabulary = np.unique(np.array(list(text)))
    text_array = string_to_array(text, vocabulary)
    return text_array, vocabulary


def transpose_seq(x, batch_size, seq_size):
    # Trim x to make its length a multiple of batch_size*seq_size
    x = x[:len(x) - len(x) % (batch_size * seq_size)]

    # Rearrange such that sequences continue over batches. This allows us to
    # propagate hidden states over gradient updates during training.
    n_batches = x.shape[0] / (seq_size*batch_size)
    x = np.reshape(x, (-1, seq_size))
    new_x = []
    for i in range(n_batches):
        new_x.append(x[i::n_batches, :])
    x = np.vstack(new_x)

    # Make batch innermost dimension
    x = np.reshape(x, (-1, batch_size, seq_size))
    x = np.transpose(x, (0, 2, 1))

    x = np.ravel(np.ascontiguousarray(x))
    return x


class SupervisedSequenceInput(dp.SupervisedInput):
    def __init__(self, x, seq_size=50, batch_size=64, epoch_size=100):
        y = x[1:]
        x = x[:-1]
        x = transpose_seq(x, batch_size, epoch_size)
        y = transpose_seq(y, batch_size, epoch_size)
        seq_batch_size = batch_size * seq_size
        super(SupervisedSequenceInput, self).__init__(x, y, seq_batch_size)
        self.epoch_size = epoch_size
        # Set random start index
        self.epoch_idx = np.random.randint(self.n_batches//self.epoch_size)
        self.n_batches = epoch_size

    def _batch_slices(self):
        for b in range(self.epoch_size):
            start = (self.epoch_idx * self.epoch_size + b) * self.batch_size
            start = min(start % self.n_samples,
                        self.n_samples - self.batch_size)
            stop = start + self.batch_size
            yield start, stop
        self.epoch_idx += 1


def new_model(n_classes, n_layers, n_hidden):
#    filler = dp.UniformFiller(low=-0.05, high=0.05)
    filler = dp.AutoFiller(gain=1.25)
    def rnn_node():
        return GatedRecurrent(n_hidden=n_hidden, w_x=filler, w_h=filler)
    recurrent_nodes = [rnn_node() for _ in range(n_layers)]
    fc_out = dp.FullyConnected(n_out=n_classes, weights=dp.AutoFiller())
    return recurrent_nodes, fc_out


def train_network(model, x_train, n_epochs=1000, learn_rate=0.2, batch_size=64,
                  seq_size=50, epoch_size=100):
    recurrent_nodes, fc_out = model
    n_classes = fc_out.n_out
    recurrent_graph = RecurrentGraph(
        recurrent_nodes=recurrent_nodes, seq_size=seq_size,
        batch_size=batch_size, cyclic=True, dropout=0.5
    )
    net = dp.NeuralNetwork(
        layers=[
            OneHot(n_classes=n_classes),
            Reshape((seq_size, batch_size, -1)),
            recurrent_graph,
            Reshape((seq_size*batch_size, -1)),
            fc_out,
        ],
        loss=dp.SoftmaxCrossEntropy(),
    )
    net.phase = 'train'

    # Prepare network inputs
    train_input = SupervisedSequenceInput(
        x_train, seq_size=seq_size, batch_size=batch_size,
        epoch_size=epoch_size
    )

    # Train network
    try:
        trainer = dp.StochasticGradientDescent(
            max_epochs=n_epochs, min_epochs=n_epochs,
            learn_rule=dp.RMSProp(learn_rate=learn_rate),
        )
        test_error = None
        trainer.train(net, train_input, test_error)
    except KeyboardInterrupt:
        pass
    return recurrent_nodes, fc_out


def test_network(model):
    seq_size = 1
    batch_size = 1
    recurrent_nodes, fc_out = model
    n_classes = fc_out.n_out
    recurrent_graph = RecurrentGraph(
        recurrent_nodes=recurrent_nodes, seq_size=seq_size,
        batch_size=batch_size, cyclic=True, dropout=0.5
    )
    net = dp.NeuralNetwork(
        layers=[
            OneHot(n_classes=n_classes),
            Reshape((seq_size, batch_size, -1)),
            recurrent_graph,
            Reshape((seq_size*batch_size, -1)),
            fc_out,
            dp.Softmax(),
        ],
        loss=dp.loss.Loss(),
    )
    net._setup(x_shape=(seq_size,))
    net.phase = 'test'
    return net


def test_error(model, x):
    y = x[1:]
    x = x[:-1]
    net = test_network(model)
    y_pred = np.zeros_like(x)
    for i in range(x.shape[0]):
        y_pred[i] = one_hot_decode(np.array(net.fprop(x=ca.array(x[i]))))
    return np.mean(y_pred != y)


def sample(model, n_samples, temp, init_str='\n'):
    net = test_network(model)
    init_str = string_to_array(init_str, vocabulary)
    for i in range(init_str.shape[0]):
        x = init_str[i]
        next_x = np.array(net.fprop(x=ca.array(x)))
    y = []
    for i in range(n_samples):
        if temp < 1:
            next_x = np.exp(np.log(next_x + 1e-10) / temp)
        prob = np.squeeze(next_x).astype(float)
        prob /= np.sum(prob)
        next_x = np.random.choice(prob.size, p=prob)
        next_x = np.array(net.fprop(x=ca.array(next_x)))
        y.append(next_x)
    y = np.vstack(y)    
    y = one_hot_decode(y)
    y = array_to_string(y, vocabulary)
    return y


x, vocabulary = dataset('shakespeare.txt')

# Split dataset
n_test = 500
x_train = x[:-n_test]
x_test = x[-n_test:]

model_path = 'model.pickle'
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = new_model(n_classes=len(vocabulary), n_layers=2, n_hidden=256)

train_network(model, x_train, seq_size=10, epoch_size=10, learn_rate=0.2, batch_size=64)
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print('Test error: %.4f' % test_error(model, x_test))

generated = sample(model, 1000, temp=0.3, init_str='thy ')
print(generated)
