import threading
import weakref


_lock = threading.RLock()
_codecs = weakref.WeakKeyDictionary()


def _codec(graph):
    with _lock:
        codec = _codecs.get(graph)
        if codec is None:
            nodes = list(graph.nodes())
            codec = ({node: index for index, node in enumerate(nodes)}, nodes)
            _codecs[graph] = codec
        return codec


def encode_node(graph, node):
    """Return the stable graph-local integer code for a node ID."""
    return _codec(graph)[0][node]


def decode_node(graph, code):
    """Return the original node ID for a graph-local integer code."""
    return _codec(graph)[1][code]