import pickle
import torch


def load_parameters(mdl, fname):
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')  # 权重字典

    own_state = mdl.state_dict()  # collections.OrderDict
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dim '
                                   'in your model are {} and whose dim in the checkpoint '
                                   'are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected Key "{}" in state_dict'.format(name))
