from typing import Union, NewType, List, Any, Callable, Optional
from .unidimensional import Func1D
import numpy as np


class FuncMultiD:
    """
    bla bla bla
    """

    def __init__(self, list_f: List[Func1D], op: str = "sum", **kwargs):
        super(FuncMultiD, self).__init__(**kwargs)
        # assess that every function lies in the same interval
        self.x_min = np.array([f.x_min for f in list_f])
        self.x_max = np.array([f.x_max for f in list_f])
        self.list_f = list_f
        self.n_dim = len(self.list_f)

        self.op = op.lower()

        if self.op not in ["max", "min", "sum", "mean"]:
            raise ValueError(
                "unknown composite operator op={} not in [{}, {}, {}, {}]".format(
                    self.op, "max", "min", "sum", "mean"
                )
            )

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # assess x \in [x_min, x_max]
        x = np.array(x)
        if len(x.shape) == 1:
            x = x[None]
        if x.shape[-1] != self.n_dim:
            raise ValueError(
                "wrong shape, expected en array of shape [None, {}]".format(self.n_dim)
            )
        output = np.array([self.list_f[i](x[:, i]) for i in range(self.n_dim)]).T

        if self.op=="sum":
            return np.sum(output, -1)[:, None]
        if self.op=="mean":
            return np.mean(output, -1)[:, None]
        if self.op=='max':
            return np.max(output, -1)[:, None]
        if self.op=="min":
            return np.min(output, -1)[:,None]
        return np.sum(output, -1)[:, None]


    def lipschitz_approx(self, p: int = np.inf, n_: int = 0) -> float:
        grad_ = [f.lipschitz_approx(p, n_) for f in self.list_f]

        if p == 1:
            return np.max(grad_)
        if p == np.inf:
            return np.sum(grad_)
        if p == 2:
            return np.sqrt(np.sul([e ** 2 for e in grad_]))

    def lipschitz(self, p: int = np.inf) -> float:

        # only an upper bound
        lip_const = [f.lipschitz(p) for f in self.list_f]

        if p == 1:
            return np.max(lip_const)
        if p == np.inf:
            return np.sum(lip_const)
        if p == 2:
            return np.sqrt(np.sul([e ** 2 for e in lip_const]))
