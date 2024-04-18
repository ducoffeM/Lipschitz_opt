from typing import Union, NewType, List, Any, Callable, Optional
import numpy as np
from heapq import heappush, heappop, heapify
from lipschitz_opt.doo.core import get_bound_box, split_midpoint, Box, get_split
import time as clock


class Node:
    """
    Representation of the bounds of a function in a domain (a Box)
    """

    def __init__(
        self, box: Box, value: Union[float, np.ndarray], maximize: bool = True
    ):
        self.box = box
        self.value = value
        self.maximize = maximize

    def __lt__(self, other: float) -> bool:
        # invert in case we are maximizing
        if self.maximize:
            return self.value > other.value
        return False

    def get_centroid(self):
        return self.box.get_midpoint()

    def update_value(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        lipschitz: float,
        ratio: Union[int, float],
    ):
        if self.value == np.inf:
            self.value = np.max(
                get_bound_box(func, lipschitz, self.box, ratio, self.maximize)
            )

    # Callable[[Union[Box, List[Box]])],List[Box]]
    def split(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        lipschitz: float,
        ratio: Union[int, float] = 1,
        split_method=None,
    ) -> List:

        if split_method is None:
            boxes = self.box.split()
        else:
            boxes = split_method(self.box)

        bounds = get_bound_box(func, lipschitz, boxes, ratio, self.maximize)

        if self.maximize:
            bounds = np.minimum(self.value, bounds)
        else:
            bounds = np.maximum(self.value, bounds)

        # create a list of Nodes
        return [
            Node(box_i, bound_i, self.maximize)
            for (box_i, bound_i) in zip(boxes, bounds)
        ]

    def get_dim(self) -> int:
        return self.box.get_dim()


def get_ratio(p: int, dim: Union[int, float]):
    if p == np.inf:
        return 1
    return dim


def doo(
    x_min,
    x_max,
    func,
    lipschitz,
    p,
    maximize=True,
    splitting_sheme=None,
    max_iter=10,
    time=False,
):
    if time:
        start_time = clock.process_time()
        log_time = []
    x_min = np.array(x_min)
    x_max = np.array(x_max)
    if len(x_min.shape) == 0:
        x_min = x_min[None]
    if len(x_max.shape) == 0:
        x_max = x_max[None]
    # create a Node
    node_init = Node(Box(x_min, x_max), np.inf, maximize)
    log_x=[node_init.get_centroid()]
    log_y =[]
    dim = node_init.get_dim()
    # get ratio from p
    ratio = get_ratio(p, dim)
    node_init.update_value(func, lipschitz, ratio)

    split_func = get_split(splitting_sheme)
    # init stack
    stack = []
    heapify(stack)
    heappush(stack, node_init)
    bound = node_init.value
    log_y.append(func(log_x[0]))
    log_bounds = []
    for _ in range(max_iter-1):

        # retrieve node with the largest bound
        node_ = heappop(stack)
        #print(node_.value)
        log_bounds.append(node_.value)
        log_y.append(max(max(log_y), func(node_.get_centroid())))
        if time:
            checkpoint_time = clock.process_time()
            log_time.append(checkpoint_time - start_time)

        if np.allclose(log_y[-1], log_bounds[-1]):
            print("Global optimum reached: {}".format(log_y[-1]))
            break

        # get leaves
        nodes = node_.split(func, lipschitz, ratio, split_func)
        for node_i in nodes:
            heappush(stack, node_i)
            log_x.append(node_i.get_centroid())

    # final value
    node_final = heappop(stack)
    bound = node_final.value
    heappush(stack, node_final)
    log_bounds.append(bound)
    if time:
        checkpoint_time = clock.process_time()
        log_time.append(checkpoint_time - start_time)
    if time:
        return bound, stack, log_x, log_y, log_bounds, log_time
    else:
        return bound, stack, log_x, log_y, log_bounds
