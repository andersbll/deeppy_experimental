class DiGraph(object):
    def __init__(self):
        self._nodes = {}
        self._pred = {}
        self._succ = {}

    def nodes(self, with_attr=False):
        if with_attr:
            return self._nodes.items()
        else:
            return self._nodes.keys()

    def edges(self, nodes=None, with_attr=False):
        if nodes is None:
            nodes = self._nodes.keys()
        for n in nodes:
            for neighbor, attr in self._succ[n].items():
                if with_attr:
                    yield (n, neighbor, attr)
                else:
                    yield (n, neighbor)

    def add_node(self, n, attr=None):
        if n not in self._succ:
            self._succ[n] = {}
            self._pred[n] = {}
            self._nodes[n] = attr
        elif attr is not None:
            raise ValueError('Node exists; cannot replace attributes.')

    def add_nodes(self, nodes, with_attr=True):
        for n in nodes:
            if with_attr:
                n, attr = n
                self.add_node(n, attr)
            else:
                self.add_node(n)

    def remove_node(self, node):
        try:
            neighbors = self._succ[node]
        except KeyError:
            raise ValueError("Node is not in the graph: %s" % node)
        for u in neighbors:
            del self._pred[u][node]
        for u in self._pred[node]:
            del self._succ[u][node]
        del self._succ[node]
        del self._pred[node]
        del self._nodes[node]

    def add_edge(self, u, v, attr=None):
        self.add_node(u)
        self.add_node(v)
        self._succ[u][v] = attr
        self._pred[v][u] = attr

    def add_edges(self, edges):
        # process ebunch
        for e in edges:
            ne = len(e)
            if ne == 3:
                u, v, dd = e
            elif ne == 2:
                u, v = e
                dd = None
            else:
                raise ValueError('Edge tuple %s must have size 2 or 3.' % e)
            self.add_edge(u, v, dd)

    def remove_edge(self, u, v):
        try:
            del self._succ[u][v]
            del self._pred[v][u]
        except KeyError:
            raise ValueError('Edge %s-%s is not in graph.' % (u, v))

    def in_degree(self):
        for node, neighbors in self._pred.items():
            yield (node, len(neighbors))

    def out_degree(self):
        for node, neighbors in self._succ.items():
            yield (node, len(neighbors))

    def __contains__(self, n):
        return n in self._nodes

    def __getitem__(self, n):
        return self._succ[n]

    def __len__(self):
        return len(self._nodes)


def topological_sort(graph, nodes=None):
    if nodes is None:
        nodes = graph.nodes()
    else:
        nodes = reversed(list(nodes))

    def dfs(graph, seen, explored, v):
        seen.add(v)
        for w in graph._succ[v]:
            if w not in seen: 
                dfs(graph, seen, explored, w)
            elif w in seen and w not in explored:
                raise ValueError('Graph contains a cycle.')
        explored.insert(0, v)

    seen = set()
    explored = []
    for v in nodes:
        if v not in explored:
            dfs(graph, seen, explored, v)
    return explored


def copy(graph):
    graph_copy = DiGraph()
    graph_copy.add_nodes(graph.nodes(with_attr=True), with_attr=True)
    graph_copy.add_edges(graph.edges(with_attr=True))
    return graph_copy


def reverse(graph):
    graph_rev = copy(graph)
    graph_rev._pred, graph_rev._succ = graph_rev._succ, graph_rev._pred
    return graph_rev
