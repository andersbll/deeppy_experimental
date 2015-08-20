from ..base import Model, ParamMixin, PickleMixin, PhaseMixin

class Node(PhaseMixin, PickleMixin):
    def _setup(self, **shapes):
        pass

    def fprop(self, **arrays):
        pass

    def bprop(self, **arrays):
        pass

    def out_shapes(self, **shapes):
        pass

class SupervisedBatch(Node):
    def __init__(self):
        self.name = 'input'
        pass

    def _setup(self, x_shape, y_shape):
        pass

    def fprop(self, x, y):
        return {'samples': x, 'labels': y}

    def bprop(self, samples_grad, labels_grad):
        pass

    def out_shapes(self, x_shape, y_shape):
        return {'samples': x_shape, 'labels': y_shape}
