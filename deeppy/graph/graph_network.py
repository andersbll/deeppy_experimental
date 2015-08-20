import numpy as np
import itertools
from ..base import Model, ParamMixin, PhaseMixin
from ..input import Input
from ..loss import Loss
from ..feedforward.layers import Layer
from . import digraph


class GraphNetwork(Model, PhaseMixin):
    def __init__(self, graph):
        self.input = None
        self.graph = graph
        self._initialized = False
        self._fprop_topology = None
        self._bprop_topology = None
        self._graph_rev = None

    def _setup(self, **in_shapes):
        if self._initialized:
            return

#        for node, in_degree in self.graph.in_degree():
#            print(node, in_degree)
#        for node, in_degree in self.graph.out_degree():
#            print(node, in_degree)
        # TODO: check that only one node has in-degree 0
        # TODO: check that only one node has out-degree 0 and is Loss

        # Create data structures for fprop and bprop array passing
        self._fprop_topology = digraph.topological_sort(self.graph)
        self._graph_rev = digraph.reverse(self.graph)
        edges = self._graph_rev.edges(with_attr=True)
        for node1, node2, (port1, port2) in edges:
            self._graph_rev.remove_edge(node1, node2)
            self._graph_rev.add_edge(node1, node2, (port2+'_grad', port1+'_grad'))
        self._bprop_topology = digraph.topological_sort(self._graph_rev)

        shapes = {node: {} for node in self._fprop_topology}
        shapes[self._fprop_topology[0]] = in_shapes
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

    def _update(self, **arrays):
        self.phase = 'train' 
        # Forward propagation
        inputs = {node: {} for node in self._fprop_topology}
        inputs[self._fprop_topology[0]] = arrays
        for node1 in self._fprop_topology[:-1]:
            out = node1.fprop(**inputs[node1])
            if isinstance(node1, Layer):
                out = {'y': out}
            for node2, (port1, port2) in self.graph[node1].items():
                inputs[node2][port2] = out[port1]

        loss_node = self._bprop_topology[0]
        grad = loss_node.grad(**inputs[loss_node])

        # Backward propagation
        grads = {node: {} for node in self._bprop_topology[1:]}
        for node2, (port1, port2) in self._graph_rev[loss_node].items():
            grads[node2][port2] = grad
        for node1 in self._bprop_topology[1:-1]:
            grad = node1.bprop(**grads[node1])
            if isinstance(node1, Layer):
                grad = {'x_grad': grad}
            for node2, (port1, port2) in self._graph_rev[node1].items():
                grads[node2][port2] = grad[port1]

        return loss_node.loss(**inputs[loss_node])

    def fprop(self, **arrays):
        inputs = {node: {} for node in self._fprop_topology}
        inputs[self._fprop_topology[0]] = arrays
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
        return out

    def out_shapes(self, **in_shapes):
        shapes = {node: {} for node in self._fprop_topology}
        shapes[self._fprop_topology[0]] = in_shapes
        for node1 in self._fprop_topology[:-1]:
            in_shapes = shapes[node1]
            if isinstance(node1, Layer):
                out_shapes = {'y': node1.y_shape(**in_shapes)}
            else:
                out_shapes = node1.out_shapes(**in_shapes)
            for node2, (port1, port2) in self.graph[node1].items():
                shapes[node2][port2 + '_shape'] = out_shapes[port1]
        last_node = self._fprop_topology[-1]
        if isinstance(last_node, Layer):
            out_shape = {'y': last_node.y_shape(**shapes[last_node])}
        else:
            out_shape = last_node.out_shapes(**shapes[last_node])
        return out_shape
