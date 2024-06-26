from .train_gcn import train_gcn
from .train_gcn_reid import train_gcn_reid

__factory__ = {
    'train_gcn': train_gcn,
    'reid_gcn': train_gcn_reid,
}


def build_handler(phase, model):
    key_handler = '{}_{}'.format(phase, model)
    if key_handler not in __factory__:
        raise KeyError("Unknown op:", key_handler)
    return __factory__[key_handler]
