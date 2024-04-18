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


def piyavskii_loop(
    x_, y_, lip, x_min, x_max, p=np.inf, maximize=True, solver=None, warm_start={}, relax=True
):
    if p == 2:
        raise NotImplementedError()
    if not p in [1, np.inf]:
        raise NotImplementedError()

    if solver is None:
        solver = pywraplp.Solver.CreateSolver("SCIP")

    n = len(x_)
    n_dim = len(x_min)


    if 'x' not in warm_start.keys():
        x = [solver.NumVar(lb=x_min[i], ub=x_max[i], name="x_{}".format(i)) for i in range(n_dim)]
        warm_start['x']=x
    else:
        x = warm_start['x']

    # x[i]- x_[i]
    for j in range(n):
        if "delta_{}".format(j) not in warm_start.keys():
            delta_j = [solver.NumVar(lb=x_min[i] - x_[j][i], ub=x_max[i] - x_[j][i], name="delta_{}_{}".format(j, i)
            ) for i in range(n_dim)]
            #warm_start["delta_{}".format(j)]=delta_j
            for i in range(n_dim):
                solver.Add(delta_j[i]<=x[i] - x_[j][i])

        abs_=[]
        if "abs_{}".format(j) not in warm_start.keys():
            abs_j = [solver.NumVar(
                lb=np.maximum(0, x_min - x_[j])[i],
                ub=np.maximum(x_max - x_[j], x_[j] - x_min)[i],
                name="abs_{}_{}".format(j, i),
            ) for i in range(n_dim)]
            abs_.append(abs_j)
            #warm_start["abs_{}".format(j)] = abs_j

            # micro souci a regler ....
            alpha_j = [solver.IntVar(0, 1, name='alpha_{}_{}'.format(j, i)) for i in range(n_dim)]
            beta_j = [solver.IntVar(0, 1, name='beta_{}_{}'.format(j, i)) for i in range(n_dim)]
            opp_delta_j = [solver.NumVar(lb=-x_max[i] + x_[j][i], ub=-x_min[i] + x_[j][i], name="delta_neg_{}_{}".format(j, i)
            ) for i in range(n_dim)]

            for i in range(n_dim):
                solver.Add(alpha_j[i]+beta_j[i]==1)
                solver.Add(delta_j[i]+opp_delta_j[i]==0)


                toto = biproduct(solver, alpha_j[i], delta_j[i], x_min[i] - x_[j][i], x_max[i] - x_[j][i], 'pos_{}_{}'.format(j,i))
                tata = biproduct(solver, beta_j[i], opp_delta_j[i],-x_max[i] + x_[j][i], -x_min[i] + x_[j][i], 'neg_{}_{}'.format(j,i))
                solver.Add(abs_j[i] == toto+tata)
                solver.Add(abs_j[i]>=delta_j[i])
                solver.Add(abs_j[i]>=-delta_j[i])



    #abs_ = [warm_start["abs_{}".format(j)] for j in range(n) ]

    if not p in [1, np.inf]:
        raise NotImplementedError()

    if "obj" in warm_start.keys():
        obj = warm_start["obj"]
    else:
        obj = solver.NumVar(lb=-np.inf, ub=np.inf, name="obj")
        warm_start["obj"] = obj


    if p == np.inf:

        for j in range(n):
            if "linf_norm_{}".format(j) not in warm_start.keys():
                z_norm_j = solver.NumVar(
                    lb=0.0,
                    ub=np.max(np.maximum(x_max - x_[j], x_[j] - x_min)),
                    name="linf_norm_{}".format(j),
                )
                #warm_start["linf_norm_{}".format(j)] = z_norm_j

                if maximize:
                    solver.Add(obj<= y_[j] + lip * z_norm_j)
                else:
                    solver.Add(obj >= y_[j] - lip * z_norm_j)

            if "biprod_{}".format(j) not in warm_start.keys():
                coeff_j = [solver.IntVar(
                    lb=0, ub=1, name="coeff_{}_{}".format(j,i)
                ) for i in range(n_dim)]
                solver.Add(sum(coeff_j)==1)
                upper_bound = [
                    np.maximum(x_max - x_[k], x_[k] - x_min) for k in range(n)
                ]
                biprod_j = [solver.NumVar(lb=0.0, ub=upper_bound[j][i], name="biprod_{}_{}".format(j, i)
                ) for i in range(n_dim)]
                #warm_start["biprod_{}".format(j)] = biprod_j
                solver.Add(z_norm_j==sum(biprod_j))

                for i in range(n_dim):
                    solver.Add(-biprod_j[i] + upper_bound[j][i] * coeff_j[i] + abs_[j][i]<=upper_bound[j][i])
                    solver.Add(biprod_j[i]<=upper_bound[j][i] * coeff_j[i])
                    solver.Add(biprod_j[i]<=abs_[j][i])
                    solver.Add(abs_[j][i]<=z_norm_j)

    if p == 1:

        for j in range(n):
            if "l1_norm_{}".format(j) not in warm_start.keys():
                z_norm_j = solver.NumVar(
                    lb=0.0,
                    ub=np.sum(np.maximum(x_max - x_[j], x_[j] - x_min)),
                    name="l1_norm_{}".format(j),
                )
                #warm_start["l1_norm_{}".format(j)] = z_norm_j

                solver.Add(z_norm_j==sum(abs_[j]),
                    name="const_norm_{}".format(j),
                )

                if maximize:
                    solver.Add(obj <=y_[j][0] + lip * z_norm_j
                    )
                else:
                    solver.Add(obj>=y_[j][0] - lip * z_norm_j
                    )
    if maximize:
        solver.Maximize(obj)
    else:
        solver.Minimize(obj)


    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        # retrieve x and compute the value of the function
        x_opt = np.array([x_j.solution_value() for x_j in x])
        # t_opt = t.solution_value()
        return x_opt, warm_start
    else:
        raise ValueError("problem...")


