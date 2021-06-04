from typing import Union, NewType, List, Any, Callable, Optional

# import numpy as np
from abc import ABC
import autograd.numpy as np
from autograd import grad as gradient

# classe abstraite for every 1D function
class Func1D(ABC):
    def __init__(self, x_min: float = 0.0, x_max: float = 1.0, n_sample: int = 1000):

        self.set_domain(x_min, x_max)
        self.n_sample = n_sample

    def set_domain(self, x_min: float, x_max: float):
        if x_min >= x_max:
            raise ValueError(
                "undefined domain of study: x_min={}>=x_max={}".format(x_min, x_max)
            )
        self.x_min = x_min
        self.x_max = x_max

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        raise NotImplementedError()

    def grad(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        if np.min(x) < self.x_min or np.max(x) > self.x_max:
            raise ValueError(
                "samples should lie in the interval [{},{}]".format(
                    self.x_min, self.x_max
                )
            )
        f_ = self.__call__

        try:
            return gradient(f_)(x)
        except TypeError:
            # split into samples
            return np.array([gradient(f_)(x_i) for x_i in x])

    def lipschitz(self, p: int = np.inf, n_: int = 0) -> float:
        if n_ == 0:
            n_ = int(np.floor(self.n_sample * (self.x_max - self.x_min)))
        alpha_ = np.linspace(0.0, 1.0, n_)
        x_ = self.x_min * alpha_ + self.x_max * (1 - alpha_)
        grad_x_ = np.abs([self.grad(x_i) for x_i in x_])

        if p == np.inf or p == 1:
            return np.max(np.abs(grad_x_))
        if p == 2:
            return np.max(grad_x_ ** 2)
        else:
            raise NotImplementedError("unimplemented Lp norm: p ={}".format(p))


class Func1D_0(Func1D):
    """
    bla bla bla

    f(x) = d - c*abs(x-a)**b
    """

    def __init__(
        self, a: float = 1.0, b: float = 1.0, c: float = 1.0, d: float = 1.0, **kwargs
    ):
        super(Func1D_0, self).__init__(**kwargs)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        if self.b < 1:
            raise NotImplementedError("Error b={}<1".format(b))

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # assess x \in [x_min, x_max]
        return self.d - self.c * np.abs(x - self.a) ** self.b


class Func1D_composite(Func1D):
    """
    bla bla bla
    """

    def __init__(self, list_f: List[Func1D_0], op: str = "max", **kwargs):
        super(Func1D_composite, self).__init__(**kwargs)
        # assess that every function lies in the same interval
        x_min_list = [f.x_min for f in list_f]
        x_max_list = [f.x_max for f in list_f]
        if min(x_min_list) != max(x_min_list):
            raise ValueError(
                "all used functions should be defined on the same interval but found x_min of function {} strictly lower than x_min of function {}".format(
                    np.argmin(x_min_list), np.argmax(x_min_list)
                )
            )
        if min(x_max_list) != max(x_max_list):
            raise ValueError(
                "all used functions should be defined on the same interval but found x_max of function {} strictly lower than x_max of function {}".format(
                    np.argmin(x_max_list), np.argmax(x_max_list)
                )
            )

        self.x_min = x_min_list[0]
        self.x_max = x_max_list[0]
        self.list_f = list_f

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
        output = np.array([f(x) for f in self.list_f])
        if self.op == "max":
            return np.max(output, 0)
        if self.op == "min":
            return np.min(output, 0)
        if self.op == "sum":
            return np.sum(output, 0)
        if self.op == "mean":
            return np.mean(output, 0)
        else:
            raise NotImplementedError("unknown operator op={}".format(self.op))

    def lipschitz(self, p: int = np.inf, n_: int = 0) -> float:
        if n_ == 0:
            n_ = int(
                np.floor(self.n_sample * (self.x_max - self.x_min)) * len(self.list_f)
            )
        return super(Func1D_composite, self).lipschitz(p=p, n_=n_)
