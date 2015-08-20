import itertools
from copy import deepcopy, copy
import cudarray as ca
import numpy as np

import deeppy as dp
from deeppy.graph.graph_network import GraphNetwork
from deeppy.graph.nodes import Node
import deeppy.graph.digraph as digraph
from deeppy.feedforward.layers import Layer
from deeppy.base import ParamMixin, PhaseMixin
from deeppy.parameter import Parameter
from deeppy.filler import Filler
from deeppy.feedforward.activation_layers import Activation


class Graph(Layer, ParamMixin, PhaseMixin):
    def __init__(self, graph, in_node, out_node):
        self.input = None
        self.graph = graph
        self.in_node = in_node
        self.out_node = out_node
        self._initialized = False
        self._fprop_topology = None
        self._bprop_topology = None
        self._graph_rev = None

    def _setup(self, x_shape):
        if self._initialized:
            return

        # Create data structures for fprop and bprop array passing
        order = []
        for node in self.graph.nodes():
            if node not in [self.in_node, self.out_node]:
                order.append(node)
        order = [self.in_node] + order + [self.out_node]

        self._fprop_topology = digraph.topological_sort(self.graph, order)

        self._graph_rev = digraph.reverse(self.graph)
        edges = self._graph_rev.edges(with_attr=True)
        for node1, node2, (port1, port2) in edges:
            self._graph_rev.remove_edge(node1, node2)
            self._graph_rev.add_edge(node1, node2, (port2+'_grad', port1+'_grad'))
        self._bprop_topology = digraph.topological_sort(self._graph_rev, reversed(order))

        shapes = {node: {} for node in self._fprop_topology}
        shapes[self.in_node]['x_shape'] = x_shape
        for node1 in self._fprop_topology:
            in_shapes = shapes[node1]
            
            node1._setup(**in_shapes)
            if node1 is self._fprop_topology[-1]:
                break
            if isinstance(node1, Layer):
                out_shapes = {'y': node1.y_shape(**in_shapes)}
            else:
                out_shapes = node1.out_shapes(**in_shapes)
            for node2, (port1, port2) in self.graph[node1].items():
                shapes[node2][port2 + '_shape'] = out_shapes[port1]
        self._initialized = True

    @property
    def _params(self):
        all_params = [node._params for node in self.graph.nodes()
                      if isinstance(node, ParamMixin)]
        # Concatenate lists in list
        return list(itertools.chain.from_iterable(all_params))

    @PhaseMixin.phase.setter
    def phase(self, phase):
        if self._phase == phase:
            return
        self._phase = phase
        for node in self.graph.nodes():
            if isinstance(node, PhaseMixin):
                node.phase = phase

    def fprop(self, x):
        inputs = {node: {} for node in self._fprop_topology}
        inputs[self.in_node]['x'] = x
        for node1 in self._fprop_topology[:-1]:
            out = node1.fprop(**inputs[node1])
            if isinstance(node1, Layer):
                out = {'y': out}
            for node2, (port1, port2) in self.graph[node1].items():
                inputs[node2][port2] = out[port1]
        last_node = self._fprop_topology[-1]
        out = last_node.fprop(**inputs[last_node])
        if isinstance(node1, Layer):
            out = {'y': out}
        return out['y']
    
    def bprop(self, y_grad):
        grads = {node: {} for node in self._bprop_topology}
        grads[self.out_node]['y_grad'] = y_grad
        for node1 in self._bprop_topology[:-1]:
            grad = node1.bprop(**grads[node1])
            if isinstance(node1, Layer):
                grad = {'x_grad': grad}
            for node2, (port1, port2) in self._graph_rev[node1].items():
                grads[node2][port2] = grad[port1]
        if isinstance(self.in_node, Layer):
            grad = {'x_grad': self.in_node.bprop(**grads[self.in_node])}
        else:
            grad = self.in_node.bprop(**grads[self.in_node])
        return grad['x_grad']

    def y_shape(self, x_shape):
        shapes = {node: {} for node in self._fprop_topology}
        shapes[self.in_node]['x_shape'] = x_shape
        for node1 in self._fprop_topology[:-1]:
            in_shapes = shapes[node1]
            if isinstance(node1, Layer):
                out_shapes = {'y': node1.y_shape(**in_shapes)}
            else:
                out_shapes = node1.out_shapes(**in_shapes)
            for node2, (port1, port2) in self.graph[node1].items():
                shapes[node2][port2 + '_shape'] = out_shapes[port1]
        if isinstance(self.out_node, Layer):
            out_shape = {'y': self.out_node.y_shape(**shapes[self.out_node])}
        else:
            out_shape = self.out_node.out_shapes(**shapes[self.out_node])
        return out_shape['y']


class OneHot(Layer):
    def __init__(self, n_classes):
        self.name = 'onehot'
        self.n_classes = n_classes

    def _setup(self, x_shape):
        pass

    def fprop(self, x):
        return ca.nnet.one_hot_encode(x, self.n_classes)

    def bprop(self, y_grad):
        raise NotImplementedError()

    def y_shape(self, x_shape):
        return x_shape + (self.n_classes,)


class Reshape(Layer):
    def __init__(self, shape):
        self.name = 'reshape'
        self.x_shape = None
        self.shape = shape

    def fprop(self, x):
        self.x_shape = x.shape
        return ca.reshape(x, self.shape)

    def bprop(self, y_grad):
        return ca.reshape(y_grad, self.x_shape)

    def y_shape(self, x_shape):
        return np.reshape(np.empty(x_shape), self.shape).shape


class Constant(Node):
    def __init__(self, shape, val=0):
        self.name = 'constant'
        self.val = val
        self.shape = shape
        self.array = np.ones(shape)
        self.array *= val
        self.array = ca.array(self.array)

    def fprop(self):
        return {'y': self.array}

    def bprop(self, y_grad):
        pass

    def out_shapes(self):
        return {'y': self.shape}


class Latch(Node):
    def __init__(self, shape, val=0):
        self.name = 'latch'
        self.val = val
        self.shape = shape
        self.array = np.ones(shape)
        self.array *= val
        self.array = ca.array(self.array)

    def fprop(self, x):
        self.array = x
        return {}

    def bprop(self):
        return {'x_grad': ca.zeros_like(self.array)}

    def out_shapes(self, x_shape):
        return {}


class LatchOut(Node):
    def __init__(self, latch):
        self.name = 'latchout'
        self.latch = latch

    def fprop(self):
        return {'y': self.latch.array}

    def bprop(self, y_grad):
        pass

    def out_shapes(self):
        return {'y': self.latch.array.shape}


class VSplit(Node):
    def __init__(self):
        self.name = 'split'
        self.n_output = -1
        self.output_shape = ()

    def _setup(self, x_shape):
        self.n_output = x_shape[0]
        self.output_shape = x_shape[1:]

    def fprop(self, x):
        self._tmp_x_shape = x.shape
        arrays = {}
        for i in range(self.n_output):
            arrays['y%i' % i] = x[i]
        return arrays

    def bprop(self, **grads):
        grad = ca.empty(self._tmp_x_shape)
        for i in range(self.n_output):
            grad[i] = grads['y%i_grad' % i]
        return {'x_grad': grad}

    def out_shapes(self, x_shape):
        out = {}
        for i in range(self.n_output):
            out['y%i' % i] = self.output_shape
        return out


class VStack(Node):
    def __init__(self):
        self.name = 'stack'
        self.n_output = -1
        self.out_shape = ()

    def _setup(self, **shapes):
        self.n_output = len(shapes)
        self.out_shape = None
        for _, shape in shapes.items():
            if self.out_shape is None:
                self.out_shape = shape
            else:
                if self.out_shape != shape:
                    raise ValueError('all input shapes must have same dimensions')

    def fprop(self, **arrays):
        array = ca.empty((self.n_output,) + self.out_shape)
        for i in range(self.n_output):
            array[i] = arrays['x%i' % i]
        return {'y': array}

    def bprop(self, y_grad):
        grads = {}
        for i in range(self.n_output):
            grads['x%i_grad' % i] = y_grad[i]
        return grads

    def out_shapes(self, **shapes):
        return {'y': (self.n_output,) + self.out_shape}


class Recurrent(Node, ParamMixin):
    def __init__(self, n_hidden, n_out, w_xh=0, w_hh=0, w_hy=0, bias_h=0.0,
                 bias_y=0, activation='tanh'):
        self.name = 'recurrent'
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.w_xh = Parameter.from_any(w_xh)
        self.w_hh = Parameter.from_any(w_hh)

        self.w_hy = Parameter.from_any(w_hy)
        self.b_h = Parameter.from_any(bias_h)
        self.b_y = Parameter.from_any(bias_y)
        self.activation = Activation.from_any(activation)
        self._tmp_x = None

    def _setup(self, x_shape, h_shape):
        batch_size, n_in = x_shape
        self.n_in = n_in
        self.w_xh._setup((n_in, self.n_hidden))
        if not self.w_xh.name:
            self.w_xh.name = self.name + '_w_xh'
        self.w_hh._setup((self.n_hidden, self.n_hidden))
        if not self.w_hh.name:
            self.w_hh.name = self.name + '_w_hh'
        self.w_hy._setup((self.n_hidden, self.n_out))
        if not self.w_hy.name:
            self.w_hy.name = self.name + '_w_hy'
        self.b_h._setup((1, self.n_hidden))
        if not self.b_h.name:
            self.b_h.name = self.name + '_b_h'
        self.b_y._setup((1, self.n_out))
        if not self.b_y.name:
            self.b_y.name = self.name + '_b_y'

    def fprop(self, x, h):
        self._tmp_x = x
        self._tmp_h_tm1 = h

        h = (ca.dot(x, self.w_xh.array) + ca.dot(h, self.w_hh.array) +
             self.b_h.array)
        h = self.activation.fprop(h)
        y = ca.dot(h, self.w_hy.array) + self.b_y.array

        self._tmp_h = h
        self._tmp_y = y
        return {'y': y, 'h': h}

    def bprop(self, y_grad, h_grad):
        ca.dot(self._tmp_h.T, y_grad, out=self.w_hy.grad_array)
        ca.sum(y_grad, axis=0, keepdims=True, out=self.b_y.grad_array)
        h_grad = h_grad + ca.dot(y_grad, self.w_hy.array.T)

        h_grad = self.activation.bprop(h_grad)
        ca.sum(h_grad, axis=0, keepdims=True, out=self.b_h.grad_array)
        ca.dot(self._tmp_h_tm1.T, h_grad, out=self.w_hh.grad_array)
        ca.dot(self._tmp_x.T, h_grad, out=self.w_xh.grad_array)

        x_grad = ca.dot(h_grad, self.w_xh.array.T)
        h_grad = ca.dot(h_grad, self.w_hh.array.T)

        return {'x_grad': x_grad, 'h_grad': h_grad}

    @property
    def _params(self):
        return self.w_xh, self.w_hh, self.w_hy, self.b_h, self.b_y

    @_params.setter
    def _params(self, params):
        self.w_xh, self.w_hh, self.w_hy, self.b_h, self.b_y = params

    def out_shapes(self, x_shape, h_shape):
        batch_size, n_in = x_shape
        return {'y': (batch_size, self.n_out), 'h': h_shape}


class GatedRecurrent(Node, ParamMixin):
    def __init__(self, n_hidden, w_x, w_h, bias_filler=0):
        self.name = 'gru'
        self.n_hidden = n_hidden

        self.w_x = Parameter.from_any(w_x)
        self.w_h = Parameter.from_any(w_h)

        self.b_r = Parameter.from_any(bias_filler)
        self.b_u = Parameter.from_any(bias_filler)
        self.b_c = Parameter.from_any(bias_filler)

        self.act_r = Activation.from_any('sigmoid')
        self.act_u = Activation.from_any('sigmoid')
        self.act_c = Activation.from_any('tanh')

        self.clip = 5
        self._tmp_x = None

    def _setup(self, x_shape, h_shape):
        batch_size, n_in = x_shape
        self.n_in = n_in

        w_x_shape = (n_in, self.n_hidden*3)
        self.w_x._setup(w_x_shape)
        w_h_shape = (self.n_hidden, self.n_hidden*3)
        self.w_h._setup(w_h_shape)

        b_shape = (self.n_hidden, 1)
        self.b_r._setup(b_shape)
        self.b_u._setup(b_shape)
        self.b_c._setup(b_shape)

    def fprop(self, x, h):
        self._tmp_x = x
        self._tmp_h_tm1 = h

        x_stack = ca.dot(self.w_x.array.T, x.T)
        h_stack = ca.dot(self.w_h.array.T, h.T)

        n = self.n_hidden
        x_r = x_stack[:n, :]
        x_u = x_stack[n:n*2, :]
        x_c = x_stack[n*2:n*3, :]
        h_r = h_stack[:n, :]
        h_u = h_stack[n:n*2, :]
        h_c = h_stack[n*2:n*3, :]

        r = self.act_r.fprop(x_r + h_r + self.b_r.array)
        u = self.act_u.fprop(x_u + h_u + self.b_u.array)
        c = self.act_c.fprop(x_c + r*h_c + self.b_c.array)

        u = ca.ascontiguousarray(ca.transpose(u))
        c = ca.ascontiguousarray(ca.transpose(c))

        h_tp1 = 1-u
        h_tp1 *= h
        h_tp1 += u*c
        
        self._tmp_r = r
        self._tmp_u = u
        self._tmp_c = c
        self._tmp_h_c = h_c
        return {'y': h_tp1, 'h': h_tp1}

    def bprop(self, y_grad, h_grad):
        n = self.n_hidden
        h_grad = h_grad + y_grad

        c_grad = h_grad * self._tmp_u
        u_grad = h_grad * (self._tmp_c - self._tmp_h_tm1)
        h_grad *= (1 - self._tmp_u)

        c_grad = ca.ascontiguousarray(ca.transpose(c_grad))
        u_grad = ca.ascontiguousarray(ca.transpose(u_grad))

        c_grad = self.act_c.bprop(c_grad)
        ca.sum(c_grad, axis=1, keepdims=True, out=self.b_c.grad_array)

        u_grad = self.act_u.bprop(u_grad)
        ca.sum(u_grad, axis=1, keepdims=True, out=self.b_u.grad_array)

        r_grad = c_grad * self._tmp_h_c
        r_grad = self.act_r.bprop(r_grad)
        ca.sum(r_grad, axis=1, keepdims=True, out=self.b_r.grad_array)

        stack_grad = ca.empty((self.n_hidden*3, y_grad.shape[0]))
        stack_grad[:n, :] = r_grad
        stack_grad[n:n*2, :] = u_grad
        stack_grad[n*2:n*3, :] = c_grad

        ca.dot(self._tmp_x.T, stack_grad.T, out=self.w_x.grad_array)
        x_grad = ca.dot(stack_grad.T, self.w_x.array.T)

        stack_grad[n*2:n*3, :] *= self._tmp_r
        ca.dot(self._tmp_h_tm1.T, stack_grad.T, out=self.w_h.grad_array)
        h_grad += ca.dot(stack_grad.T, self.w_h.array.T)

        ca.clip(h_grad, -self.clip, self.clip, out=h_grad)
        return {'x_grad': x_grad, 'h_grad': h_grad}

    @property
    def _params(self):
        return self.w_x, self.w_h, self.b_r, self.b_u, self.b_c

    @_params.setter
    def _params(self, params):
        self.w_x, self.w_h, self.b_r, self.b_u, self.b_c = params

    def out_shapes(self, x_shape, h_shape):
        batch_size, n_in = x_shape
        return {'y': (batch_size, self.n_hidden), 'h': h_shape}



class RecurrentGraph(Graph):
    def __init__(self, recurrent_nodes, seq_size, batch_size, cyclic, dropout):
        self.recurrent_nodes = recurrent_nodes
        graph = digraph.DiGraph()
        split = VSplit()
        stack = VStack()
        recurrent_graph_nodes = []
        depth = len(recurrent_nodes)
        latches = []
        for i in range(depth):
            node = recurrent_nodes[i]
            recurrent_graph_nodes.append([])
            hidden_shape = (batch_size, node.n_hidden)
            latch = Latch(hidden_shape)
            latches.append((latch, LatchOut(latch))) 
            for j in range(seq_size):
                if j > 0:
                    orig_node = node
                    node = deepcopy(node)
                    node._params = [p.share() for p in orig_node._params]
                recurrent_graph_nodes[i].append(node)

                if i == 0:
                    graph.add_edge(split, node, ('y%i' % j, 'x'))
                else:
                    node_below = recurrent_graph_nodes[i-1][j]
                    if dropout > 0:
                        drop_layer = dp.Dropout(dropout)
                        graph.add_edge(node_below, drop_layer, ('y', 'x'))
                        node_below = drop_layer
                    graph.add_edge(node_below, node, ('y', 'x'))
                if i == depth-1:
                    graph.add_edge(node, stack, ('y', 'x%i' % j))
                
                if j == 0:
                    if cyclic:
                        graph.add_edge(latches[i][1], node, ('y', 'h'))
                    else:
                        constant = Constant(hidden_shape)
                        graph.add_edge(constant, node, ('y', 'h'))
                else:
                    node_prev = recurrent_graph_nodes[i][j-1]
                    graph.add_edge(node_prev, node, ('h', 'h'))
                if j == seq_size-1:
                    graph.add_edge(node, latches[i][0], ('h', 'x'))


        super(RecurrentGraph, self).__init__(graph, in_node=split, out_node=stack)

    @property
    def _params(self):
        all_params = [node._params for node in self.recurrent_nodes
                      if isinstance(node, ParamMixin)]
        # Concatenate lists in list
        return list(itertools.chain.from_iterable(all_params))
