from __future__ import absolute_import
import pytest
import numpy as np
from lipschitz_opt.application import Func1D_0, Func1D_composite
from numpy.testing import assert_almost_equal


@pytest.mark.parametrize(
    "a, b, c, x_min, x_max", [(0.5, 1, 0.6, 0.0, 1.0), (0.3, 3, 0.6, -1.0, 1.0)]
)
def test_gradient_simple(a, b, c, x_min, x_max):
    # consider an odd value of b

    f_ref = lambda x: -c * (x - a) ** (2 * b)

    f_ = Func1D_0(a=a, b=2 * b, c=c, d=0, x_min=x_min, x_max=x_max)

    x = np.linspace(0.0, 1.0, 100) * x_min + (1 - np.linspace(0.0, 1.0, 100)) * x_max

    y_ref = f_ref(x)
    y_ = f_(x)

    assert_almost_equal(y_ref, y_, err_msg="reconstruction error")

    f_grad_ref = lambda x: -2 * b * c * (x - a) ** (2 * b - 1)

    z_ref = f_grad_ref(x)
    z_ = f_.grad(x)

    assert_almost_equal(z_ref, z_, err_msg="reconstruction error")

    L_ref = max(np.abs(z_ref[0]), np.abs(z_ref[-1]))
    L_ = f_.lipschitz()

    assert_almost_equal(L_ref, L_, err_msg="reconstruction error")
