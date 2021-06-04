from __future__ import absolute_import
from typing import Union, NewType, List, Any, Callable, Optional
import numpy as np
from ortools.linear_solver import pywraplp


def biproduct(solver, x_, f_, l_f, u_f, name):
    """

    :param solver:
    :param x_:
    :param f_:
    :param l_f:
    :param u_f:
    :param name:
    :return:
    """
    # cast to float
    if not isinstance(l_f, float):
        l_f = np.float(l_f)
    if not isinstance(u_f, float):
        u_f = np.float(u_f)

    z_ = solver.NumVar(min(0.0, l_f), max(0.0, u_f), "biprod_z_{}".format(name))
    solver.Add(-z_ + l_f * x_ <= 0)
    solver.Add(-z_ + u_f * x_ + f_ <= u_f)
    solver.Add(z_ - u_f * x_ <= 0.0)
    solver.Add(z_ - l_f * x_ - f_ <= -l_f)

    return z_


def relu(solver, x: float, upper: float, name):
    """

    :param solver:
    :param x:
    :param upper:
    :param name:
    :return:
    """
    y = solver.IntVar(0, 1, "y_relu_{}".format(name))
    M = max(1, upper)
    h = solver.NumVar(0.0, M, "h_relu_{}".format(name))

    # add constraint
    solver.Add(h >= x)
    solver.Add(h <= M * y)
    solver.Add(h <= x + M * (1 - y))

    return h, y


def l_1_norm(solver, x, upper, lower, name):
    """
    compute ||x||_1
    :param solver:
    :param x:
    :param upper:
    :param lower:
    :return:
    """

    # max(0, x)
    h_relu_pos = [
        relu(solver, x_i, upper_i, "linf_{}_pos_{}".format(k, name))[0]
        for (k, x_i, upper_i) in zip(range(len(x)), x, upper)
    ]

    # max(0, -x)
    h_relu_neg = [
        relu(solver, -x_i, -lower_i, "linf_{}_neg_{}".format(k, name))[0]
        for (k, x_i, lower_i) in zip(range(len(x)), x, lower)
    ]

    upper_bound = np.sum(np.maximum(np.abs(upper), np.abs(lower)))  # cast to float ?
    t = solver.NumVar(0.0, upper_bound, "l1_t_{}".format(name))

    # t == sum h_relu_pos + sum h_relu_neg
    solver.Add(t == sum(h_relu_neg) + sum(h_relu_pos))

    return t


def l_inf_norm(solver, x, upper, lower, name):
    """
    compute ||x||_inf
    :param solver:
    :param x:
    :param upper:
    :param lower:
    :return:
    """

    # max(0, x)
    h_relu_pos = [
        relu(solver, x_i, upper_i, "linf_{}_pos_{}".format(k, name))[0]
        for (k, x_i, upper_i) in zip(range(len(x)), x, upper)
    ]

    # max(0, -x)
    h_relu_neg = [
        relu(solver, -x_i, -lower_i, "linf_{}_neg_{}".format(k, name))[0]
        for (k, x_i, lower_i) in zip(range(len(x)), x, lower)
    ]

    abs_x = [h_pos + h_neg for (h_pos, h_neg) in zip(h_relu_pos, h_relu_neg)]

    coeff_inf = [
        solver.IntVar(0, 1, "linf_coeff_{}_{}".format(k, name)) for k in range(len(x))
    ]

    # only one coefficient is set to 1
    solver.Add(sum(coeff_inf) == 1)

    # product coeff*|x|
    upper_abs = np.maximum(upper, -lower)
    z = [
        biproduct(solver, alpha_i, x_i, 0.0, u_i, name)
        for (alpha_i, x_i, u_i) in zip(coeff_inf, abs_x, upper_abs)
    ]

    upper_bound = np.max(upper_abs)  # cast to float ?
    t = solver.NumVar(0.0, upper_bound, "linf_t_{}".format(name))
    # t = sum(z)
    solver.Add(t == sum(z))

    for x_abs_i in abs_x:
        solver.Add(t >= x_abs_i)

    return t


def l_norm(solver, x, upper, lower, name, p):

    if p == 1:
        return l_1_norm(solver, x, upper, lower, name)
    else:
        return l_inf_norm(solver, x, upper, lower, name)


def eval_loop(x, x_i, y_i, lip, p, maximize):

    if maximize:
        c = 1
    else:
        c = -1

    if p == 1:
        return np.array(
            np.min(
                [
                    y_ij + c * lip * np.sum(np.abs(x - x_ij))
                    for (x_ij, y_ij) in zip(x_i, y_i)
                ]
            )
        )
    else:
        return np.array(
            np.min(
                [
                    y_ij + c * lip * np.max(np.abs(x - x_ij))
                    for (x_ij, y_ij) in zip(x_i, y_i)
                ]
            )
        )


def piyavskii_loop(x_i, y_i, lip, x_min, x_max, p=np.inf, maximize=True):

    if p == 2:
        raise NotImplementedError()
    if not p in [1, np.inf]:
        raise NotImplementedError()

    if len(x_i) == 1:
        # super simple, do it by hand
        f_max = eval_loop(x_max, x_i, y_i, lip, p, maximize)
        f_min = eval_loop(x_min, x_i, y_i, lip, p, maximize)
        if f_max >= f_min:
            return x_max, f_max
        else:
            return x_min, f_min

    solver = pywraplp.Solver.CreateSolver("SCIP")
    # compute norm
    n = len(x_i)

    x = [solver.NumVar(x_min[i], x_max[i], "x_{}".format(i)) for i in range(len(x_min))]
    c=1
    if not maximize:
        c = -1
    z_norm = [
        y_i[k]
        + c
        * lip
        * l_norm(
            solver,
            [x[j] - x_i[k][j] for j in range(len(x))],
            x_max - x_i[k],
            x_min - x_i[k],
            k,
            p,
        )
        for k in range(n)
    ]
    z_norm = [z_i[0] for z_i in z_norm]

    coeff_inf = [
        solver.IntVar(0, 1, "loop_coeff_{}".format(k)) for (k, z_i) in enumerate(z_norm)
    ]
    solver.Add(
        sum(coeff_inf) == 1
    )  # only one element is considered which is the smallest one

    # product coeff_inf*z_norm

    # product coeff*|x|
    if maximize:
        upper_ = np.array(
            [
                y_i[k] + lip * np.sum(np.maximum(x_max - x_i[k], x_i[k] - x_min))
                for k in range(n)
            ]
        )
        lower_ = np.array(y_i)
    else:
        upper_ = np.array(y_i)
        lower_ = np.array(
            [
                y_i[k] - lip * np.sum(np.maximum(x_max - x_i[k], x_i[k] - x_min))
                for k in range(n)
            ]
        )

    z = [
        biproduct(solver, alpha_i, z_i, l_i, u_i, k)
        for (alpha_i, z_i, l_i, u_i, k) in zip(
            coeff_inf, z_norm, lower_, upper_, range(n)
        )
    ]

    t = solver.NumVar(-np.inf, np.inf, "obj_loop")
    solver.Add(t == sum(z))

    if maximize:
        for k in range(n):
            solver.Add(t <= z_norm[k])
        # objective: maximize t
        solver.Maximize(t)
    else:
        for k in range(n):
            solver.Add(t >= z_norm[k])
        # objective: minimize t
        solver.Minimize(t)

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        # retrieve x and compute the value of the function
        x_opt = np.array([x_j.solution_value() for x_j in x])
        # t_opt = t.solution_value()
        return x_opt, np.array(eval_loop(x_opt, x_i, y_i, lip, p, maximize))
    else:
        raise ValueError("problem...")


def piyavskii(
    f, lipschitz, x_min, x_max, p=np.inf, maximize=True, n_iter=10, init_random=False
):
    """

    :param f:
    :param lipschitz:
    :param x_min:
    :param x_max:
    :param p:
    :param maximize:
    :param n_iter:
    :param init_random:
    :return:
    """

    if p not in [1, np.inf]:
        raise NotImplementedError()

    if isinstance(x_min, float):
        x_min = np.zeros((1,)) + x_min

    if isinstance(x_max, float):
        x_max = np.zeros((1,)) + x_max

    n_dim = len(x_min)
    # compute initial upper bound
    if init_random:
        alpha = np.array([np.random.rand() for _ in range(n_dim)])
        x_init = alpha * x_min + (1 - alpha) * x_max
    else:
        x_init = (x_min + x_max) / 2.0

    y_init = f(x_init)
    dist_init = np.maximum(np.abs(x_max - x_init), np.abs(x_init - x_min))
    if p == 1:
        dist_init = np.sum(dist_init)
    if p == np.inf:
        dist_init = np.max(dist_init)

    if maximize:
        upper_init = y_init + lipschitz * dist_init
    else:
        upper_init = y_init - lipschitz * dist_init

    x_ = [
        x_init
    ]  # instead of a list do an array and provide the number of components (more efficient)
    y_ = [y_init]
    upper_ = [upper_init]

    for _ in range(n_iter):
        x_k, f_hat_k = piyavskii_loop(
            x_, y_, lipschitz, x_min, x_max, p, maximize=maximize
        )
        y_k = f(x_k)
        upper_.append(f_hat_k)
        x_.append(x_k)
        y_.append(y_k)

        if np.allclose(y_k, f_hat_k):
            print("Global optimum reached: {}".format(y_k[0]))
            break

    upper_[0] = np.array(upper_[0][0])
    return [
        max(y_k[0], upper_[-1]),
        {"upper": np.array(upper_), "x": np.concatenate(x_), "y": np.concatenate(y_)},
    ]
