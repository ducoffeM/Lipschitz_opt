from typing import Union, NewType, List, Any, Callable, Optional
import six
import numpy as np

BINARY = "binary"
MIDPOINT = "midpoint"


class Box:
    """
    Representation of a hypercube by two vertices: the one with the lowest and the second with the highest coordinates.
    If the vertex is provided as a tensor of shape [1, n_dim] it is automatically flattened into a vector
    """

    def __init__(self, x_min: np.ndarray, x_max: np.ndarray):

        if len(x_min.shape) > 1 and x_min.shape[0] != 1:
            raise NotImplementedError()  # better error

        if len(x_max.shape) > 1 and x_max.shape[0] != 1:
            raise NotImplementedError()  # better error

        self.x_min = np.array(x_min.flatten(), dtype="float")
        self.x_max = np.array(x_max.flatten(), dtype="float")

        # cast

    def get_dim(self) -> int:
        """

        :return: the dimension of the associated hypercube
        """
        return np.prod(self.x_min.shape)

    def get_max_dist(self) -> int:
        """
        :return: largest L_{inf} distance between two samples in the box
        """
        return np.max(self.x_max - self.x_min)

    def get_midpoint(self) -> np.ndarray:
        return (self.x_min + self.x_max) / 2.0

    def split(self):
        return split_midpoint(self)


def split_binary(
    box: Union[Box, List[Box]], index: Optional[Union[int, List[int]]] = None
) -> List[Box]:
    """
    A way of cutting the space into sub-space: select one dimension (usually the largest) and cut the space
    into two hypercubes according to this dimension.
    :param box: either a Box or a list of Boxes
    :param index: either the dimension is provided or the one which owes the largest value is automatically picked
    :return: a List of subdomains (Boxes)
    """

    if isinstance(box, Box):
        x_min = box.x_min.flatten()
        x_max = box.x_max.flatten()

        if index is None:
            index = np.argmax(x_max - x_min)

        x_max_0 = np.copy(x_max) + 0.0
        x_min_1 = np.copy(x_min) + 0.0
        value = (x_max[index] + x_min[index]) / 2.0
        x_max_0[index] = value
        x_min_1[index] = value

        # if np.allclose(x_min, x_max_0) or np.allclose(x_min_1, x_max):
        #    import pdb; pdb.set_trace()

        return [Box(x_min, x_max_0), Box(x_min_1, x_max)]
    else:
        if not isinstance(box, list):
            raise NotImplementedError()
        n = len(box)
        order = np.arange(n)
        x_min = np.concatenate([b.x_min[None] for b in box], 0)
        x_max = np.concatenate([b.x_max[None] for b in box], 0)
        if index is None:
            index = np.argmax(x_max - x_min, -1)
        x_max_0 = np.copy(x_max) + 0.0
        x_min_1 = np.copy(x_min) + 0.0
        x_max_0[order, index] = (x_min[order, index] + x_max[order, index]) / 2.0
        x_min_1[order, index] = (x_min[order, index] + x_max[order, index]) / 2.0

        return [Box(x_min[i], x_max_0[i]) for i in range(n)] + [
            Box(x_min_1[i], x_max[i]) for i in range(n)
        ]

    return [Box(0, 0)]


def split_midpoint(box: Union[Box, List[Box]]) -> List[Box]:
    """
    Split equally along every dimensions (exponential complexity, do it only on small dimension)
    :param box: either a box or a list of boxes
    :return: a list of subdomains (boxes)
    """

    # compute the number of dimensions
    if isinstance(box, Box):
        n_dim = box.get_dim()
    else:
        n_dim = box[0].get_dim()

    dom = box

    for i in range(n_dim):
        dom = split_binary(dom, i)

    return dom


def get_bound_box(
    func: Callable[[np.ndarray], np.ndarray],
    lipschitz: Union[float, np.ndarray, List[float]],
    box: Union[Box, List[Box]],
    ratio: int = 1,
    maximize: bool = True,
) -> np.ndarray:

    # get mid_points
    if isinstance(box, Box):
        midpoints = box.get_midpoint()
        distances = box.get_max_dist() / 2.0
    else:
        midpoints = np.concatenate([box_.get_midpoint()[None] for box_ in box], 0)
        distances = np.array([box_.get_max_dist() for box_ in box]) / 2.0
        distances = distances[:, None]

    # evaluate the function func on midpoints
    value_ = func(midpoints)

    # worst case value of the function inside the domain(s)
    worst = value_
    if maximize:
        worst += lipschitz * ratio * distances
    else:
        worst -= lipschitz * ratio * distances

    return worst


def deserialize_split(name):

    name = name.lower()

    if name == BINARY:
        return split_binary
    if name == MIDPOINT:
        return split_midpoint

    raise ValueError("Could not interpret " "split function identifier:", name)


def get_split(identifier):

    if identifier is None:
        return split_binary
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        return deserialize_split(identifier)
    elif callable(identifier):

        # TO DO  check the type of inputs and output
        return identifier
    else:
        raise ValueError(
            "Could not interpret " "split function identifier:", identifier
        )
