from __future__ import absolute_import
from typing import Union, NewType, List, Any, Callable, Optional
import numpy as np
from .core_ortools import piyavskii_loop as piyav_ortools
from .core_gurobi import piyavskii_loop as piyav_gurobi_
from .core_gb import piyavskii_loop as piyav_gurobi
from .draft import model_grbpiyavskii
import time as clock

def eval_loop(x, x_i, y_i, lip, p, maximize):

    X = np.array(x_i)
    Y = np.array(y_i).flatten()
    x_hat = x.reshape((1, -1))

    if maximize:
        c=1
    else:
        c=-1

    if p not in [1,2, np.inf]:
        raise NotImplementedError()


    if p==np.inf:
        toto = np.min(Y + c*lip * np.max(np.abs(x_hat - X), -1))
    elif p==1:
        toto = np.min(Y + c*lip * np.sum(np.abs(x_hat - X), -1))
    elif p==2:
        toto = np.min(Y + lip * np.sqrt(np.sum(np.abs(x_hat - X)**2, -1)))

    return toto



def piyavskii(
    f, lipschitz, x_min, x_max, p=np.inf, maximize=True, n_iter=10, init_random=False, time=True, use_gurobi=True, relax=True, verbose=0
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

    if p not in [1, 2, np.inf]:
        raise NotImplementedError()
    if time:
        start_time = clock.process_time()
        log_time = []

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
    #if n_dim>1:
    y_init= y_init[:, 0]
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
    log_bounds = []

    global_reached = False
    if use_gurobi:
        from gurobi import Model
        solver= Model('gurobi')
    else:
        from ortools.linear_solver import pywraplp
        solver = pywraplp.Solver.CreateSolver("SCIP")

    warm_start={}
    for k in range(n_iter):
        if verbose:
            print(k)
        if not use_gurobi:
            x_k, warm_start= piyav_ortools(
                x_, y_, lipschitz, x_min, x_max, p, maximize=maximize, solver=solver, warm_start=warm_start, relax=relax,
            )
        else:
            #
            x_k = piyav_gurobi(
                x_, y_, lipschitz, x_min, x_max, p, maximize=maximize, solver=solver, relax=relax
            )
        f_hat_k = np.array(eval_loop(x_k, x_, y_, lipschitz, p, maximize=maximize))
        log_bounds.append(f_hat_k)
        y_k = f(x_k)
        #if n_dim>1:
        y_k=y_k[:,0]
        upper_.append(f_hat_k)
        x_.append(x_k)
        y_.append(y_k)
        #print(y_k, f_hat_k)

        if time:
            checkpoint_time = clock.process_time()
            log_time.append(checkpoint_time - start_time)

        if np.allclose(y_k, f_hat_k):
            print("Global optimum reached: {}".format(y_k[0]))
            break


    bound = log_bounds[-1]
    stack = {"upper": np.array(upper_), "x": x_, "y": y_}

    if time:
        return bound, stack, log_bounds, log_time
    else:
        return bound, stack, log_bounds


