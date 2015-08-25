from copy import copy
import itertools
import numpy as np
import cudarray as ca
import deeppy as dp
from deeppy.graph.recurrent_graph import Recurrent, GatedRecurrent

from test_layers import check_grad


batch_sizes = [1, 4, 5]
n_ins = [1, 2, 7, 8]
n_hiddens = [1, 2, 7, 8]

if ca._backend == 'cuda':
    rtol = 1e-4
    atol = 1e-5
else:
    rtol = 1e-6
    atol = 1e-7


class LayerWrap(dp.feedforward.layers.Layer, dp.base.ParamMixin):
    def __init__(self, node, in_port, out_port, in_arrays, grad_arrays):
        self.node = node
        self.in_port = in_port
        self.out_port = out_port
        self.in_arrays = {}
        for k, array in in_arrays.items():
            self.in_arrays[k] = ca.array(array)

        self.grad_arrays = {}
        for k, array in grad_arrays.items():
            self.grad_arrays[k + '_grad'] = ca.zeros_like(array)

    def in_shapes(self, x_shape):
        shapes = {}
        for k, v in self.in_arrays.items():
            shapes[k + '_shape'] = v.shape
        shapes[self.in_port + '_shape'] = x_shape
        return shapes

    def _setup(self, x_shape):
        in_shapes = self.in_shapes(x_shape)
        self.node._setup(**in_shapes)

    def fprop(self, x):
        in_arrays = copy(self.in_arrays)
        in_arrays[self.in_port] = x
        out = self.node.fprop(**in_arrays)
        return out[self.out_port]

    def bprop(self, y_grad):
        grad_arrays = copy(self.grad_arrays)
        grad_arrays[self.out_port + '_grad'] = y_grad
        grads = self.node.bprop(**grad_arrays)
        return grads[self.in_port + '_grad']

    def y_shape(self, x_shape):
        in_shapes = self.in_shapes(x_shape)
        out_shapes = self.node.out_shapes(**in_shapes)
        return out_shapes[self.out_port]

    @property
    def _params(self):
        return self.node._params

    @_params.setter
    def _params(self, params):
        self.node._params = params



def check_node_grads(node, in_arrays, grad_arrays, rtol=None, atol=None):
    for in_port, in_array in in_arrays.items():
        for out_port, out_array in grad_arrays.items():
#            print(in_port, out_port)
            layer = LayerWrap(node, in_port, out_port, in_arrays, grad_arrays)
            layer._setup(in_arrays[in_port].shape)
            check_grad(layer, in_arrays[in_port], rtol=rtol, atol=atol)


def test_recurrent():
    n_outs = [1, 4, 5]
    confs = itertools.product(batch_sizes, n_ins, n_hiddens, n_outs)
    for batch_size, n_in, n_hidden, n_out in confs:
        print('Recurrent: batch_size=%i, n_in=%i, n_hidden=%i, n_out=%i'
              % (batch_size, n_in, n_hidden, n_out))
        x_shape = (batch_size, n_in)
        h_shape = (batch_size, n_hidden)
        y_shape = (batch_size, n_out)

        x = np.random.normal(size=x_shape).astype(dp.float_)
        h = np.random.normal(size=h_shape).astype(dp.float_)
        y = np.random.normal(size=y_shape).astype(dp.float_)
        h_next = np.random.normal(size=h_shape).astype(dp.float_)

        node = Recurrent(
            n_hidden=n_hidden,
            n_out=n_out,
            w_xh=dp.AutoFiller(),
            w_hh=dp.AutoFiller(),
            w_hy=dp.AutoFiller(),
        )

        in_arrays = {'x': x, 'h': h}
        grad_arrays = {'y': y, 'h': h_next}

        check_node_grads(node, in_arrays, grad_arrays, rtol=rtol, atol=atol)


def test_gated_recurrent():
    confs = itertools.product(batch_sizes, n_ins, n_hiddens)
    for batch_size, n_in, n_hidden in confs:
        print('GatedRecurrent: batch_size=%i, n_in=%i, n_hidden=%i'
              % (batch_size, n_in, n_hidden))
        x_shape = (batch_size, n_in)
        h_shape = (batch_size, n_hidden)
        y_shape = (batch_size, n_hidden)
        
        x = np.random.normal(size=x_shape).astype(dp.float_)
        h = np.random.normal(size=h_shape).astype(dp.float_)
        y = np.random.normal(size=y_shape).astype(dp.float_)
        h_next = np.random.normal(size=h_shape).astype(dp.float_)

        node = GatedRecurrent(
            n_hidden=n_hidden,
            w_x=dp.AutoFiller(),
            w_h=dp.AutoFiller(),
        )

        in_arrays = {'x': x, 'h': h}
        grad_arrays = {'y': y, 'h': h_next}

        check_node_grads(node, in_arrays, grad_arrays, rtol=rtol, atol=atol)
