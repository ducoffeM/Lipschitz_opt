from __future__ import absolute_import
from typing import Union, NewType, List, Any, Callable, Optional
import numpy as np

# from ortools.linear_solver import pywraplp
from gurobi import LinExpr, Model, GRB, quicksum, GurobiError


def piyavskii_loop(
    x_, y_, lip, x_min, x_max, p=np.inf, maximize=True, solver=None, relax=True
):
    relax =False
    #if p == 2:
    #    raise NotImplementedError()
    if not p in [1, 2, np.inf]:
        raise NotImplementedError()

    if solver is None:
        solver = Model("piyavskii")

    n = len(x_)
    n_dim = len(x_min)

    if len(solver.getVars()) == 0 or solver.getVarByName("x[0]") is None:
        x = solver.addVars(n_dim, lb=x_min, ub=x_max, name="x")
        solver.update()
    else:

        x = [solver.getVarByName("x[{}]".format(i)) for i in range(n_dim)]

    # x[i]- x_[i]
    for j in range(n):
        if not solver.getVarByName("delta_{}[0]".format(j)):
            delta_j = solver.addVars(
                n_dim, lb=x_min - x_[j], ub=x_max - x_[j], name="delta_{}".format(j)
            )
            for i in range(n_dim):
                solver.addConstr(
                    lhs=delta_j[i],
                    rhs=x[i] - x_[j][i],
                    sense=GRB.EQUAL,
                    name="const_delta_{}_{}".format(j, i),
                )

        if p!=2 and not solver.getVarByName("abs_{}[0]".format(j)):
            print('kiikou')
            abs_j = solver.addVars(
                n_dim,
                lb=np.maximum(0, x_min - x_[j]),
                ub=np.maximum(x_max - x_[j], x_[j] - x_min),
                name="abs_{}".format(j),
            )

            for i in range(n_dim):
                solver.addGenConstrAbs(
                    abs_j[i], delta_j[i], name="const_abs_{}_{}".format(j, i)
                )
        solver.update()

    if p!=2:
        abs_ = [
            [solver.getVarByName("abs_{}[{}]".format(j, i)) for i in range(n_dim)]
            for j in range(n)
        ]


    obj = solver.getVarByName("obj")
    if not obj:
        obj = solver.addVar(lb=-np.inf, ub=np.inf, name="obj")

    if p == np.inf:

        for j in range(n):
            if not solver.getVarByName("linf_norm_{}".format(j)):
                z_norm_j = solver.addVar(
                    lb=0.0,
                    ub=np.max(np.maximum(x_max - x_[j], x_[j] - x_min)),
                    name="linf_norm_{}".format(j),
                )

                if maximize:
                    solver.addConstr(
                        lhs=obj, rhs=y_[j][0] + lip * z_norm_j, sense=GRB.LESS_EQUAL
                    )
                else:
                    solver.addConstr(
                        rhs=obj, lhs=y_[j][0] - lip * z_norm_j, sense=GRB.LESS_EQUAL
                    )

            if not solver.getVarByName("biprod_{}[0]".format(j)):
                coeff_j = solver.addVars(
                    n_dim, lb=0, ub=1, vtype=GRB.BINARY, name="coeff_{}".format(j)
                )
                solver.addConstr(lhs=quicksum(coeff_j), rhs=1, sense=GRB.EQUAL)
                upper_bound = [
                    np.maximum(x_max - x_[k], x_[k] - x_min) for k in range(n)
                ]
                biprod_j = solver.addVars(
                    n_dim, lb=0.0, ub=upper_bound[j], name="biprod_{}".format(j)
                )
                solver.addConstr(
                    lhs=z_norm_j, rhs=quicksum(biprod_j), sense=GRB.EQUAL
                )

                for i in range(n_dim):
                    solver.addConstr(
                        lhs=-biprod_j[i] + upper_bound[j][i] * coeff_j[i] + abs_[j][i],
                        rhs=upper_bound[j][i],
                        sense=GRB.LESS_EQUAL,
                    )
                    solver.addConstr(
                        lhs=biprod_j[i],
                        rhs=upper_bound[j][i] * coeff_j[i],
                        sense=GRB.LESS_EQUAL,
                    )
                    solver.addConstr(
                        lhs=biprod_j[i], rhs=abs_[j][i], sense=GRB.LESS_EQUAL
                    )
                    solver.addConstr(
                        lhs=abs_[j][i], rhs=z_norm_j, sense=GRB.LESS_EQUAL
                    )

    if p == 1:

        for j in range(n):
            if not solver.getVarByName("l1_norm_{}".format(j)):
                z_norm_j = solver.addVar(
                    lb=0.0,
                    ub=np.sum(np.maximum(x_max - x_[j], x_[j] - x_min)),
                    name="l1_norm_{}".format(j),
                )

                solver.addConstr(
                    lhs=z_norm_j,
                    rhs=quicksum(abs_[j]),
                    sense=GRB.EQUAL,
                    name="const_norm_{}".format(j),
                )

                if maximize:
                    solver.addConstr(
                        lhs=obj, rhs=y_[j][0] + lip * z_norm_j, sense=GRB.LESS_EQUAL
                    )
                else:
                    solver.addConstr(
                        rhs=obj, lhs=y_[j][0] - lip * z_norm_j, sense=GRB.LESS_EQUAL
                    )

    if p==2:
        if not relax:
            solver.params.NonConvex=2

            for j in range(n):
                if not solver.getVarByName("l2_norm_{}".format(j)):
                    u_j = np.maximum((x_max-x_[j])**2, (x_min-x_[j])**2)
                    z_norm_j = solver.addVar(
                        lb=0,
                        ub=np.sqrt(np.sum(u_j)),
                        name="l2_norm_{}".format(j),
                    )

                    b_j = solver.addVars(n_dim, lb=0, ub=u_j, name='l2_norm_b_{}_{}'.format(j, i))
                    solver.addConstr(lhs = z_norm_j*z_norm_j, rhs=quicksum(b_j), sense=GRB.EQUAL)

                    for i in range(n_dim):
                        solver.addConstr(lhs=delta_j[i] * delta_j[i], rhs=b_j[i], sense=GRB.EQUAL)

                    if maximize:
                        solver.addConstr(
                            lhs=obj, rhs=y_[j][0] + lip * z_norm_j, sense=GRB.LESS_EQUAL
                        )
                    else:
                        solver.addConstr(
                            rhs=obj, lhs=y_[j][0] - lip * z_norm_j, sense=GRB.LESS_EQUAL
                        )
        else:
            # use McCormick's envelope to linearize the quadratic part
            for j in range(n):
                if not solver.getVarByName("l2_norm_{}".format(j)):
                    u_j = np.maximum((x_max-x_[j])**2, (x_min-x_[j])**2)
                    z_norm_j = solver.addVar(
                        lb=0,
                        ub=np.sqrt(np.sum(u_j)),
                        name="l2_norm_{}".format(j),
                    )

                    b_j = solver.addVars(n_dim, lb=0, ub=u_j, name='l2_norm_b_{}_{}'.format(j, i))
                    solver.addConstr(rhs=quicksum(b_j),lhs=2*np.sqrt(np.sum(u_j))*z_norm_j - np.sum(u_j), sense=GRB.LESS_EQUAL)
                    solver.addConstr(lhs=quicksum(b_j), rhs=np.sqrt(np.sum(u_j))*z_norm_j, sense=GRB.LESS_EQUAL)

                    lower_j = x_min - x_[j]
                    upper_j = x_max - x_[j]

                    for i in range(n_dim):
                        solver.addConstr(rhs=b_j[i], lhs=2*lower_j[i]*delta_j[i] - lower_j[i]**2, sense=GRB.LESS_EQUAL)
                        solver.addConstr(rhs=b_j[i], lhs=2 * upper_j[i] * delta_j[i] - upper_j[i] ** 2,
                                         sense=GRB.LESS_EQUAL)
                        solver.addConstr(lhs=b_j[i], rhs=(upper_j[i]+lower_j[i])*delta_j[i] - upper_j[i]*lower_j[i], sense=GRB.LESS_EQUAL)

                    if maximize:
                        solver.addConstr(
                            lhs=obj, rhs=y_[j][0] + lip * z_norm_j, sense=GRB.LESS_EQUAL
                        )
                    else:
                        solver.addConstr(
                            rhs=obj, lhs=y_[j][0] - lip * z_norm_j, sense=GRB.LESS_EQUAL
                        )


    if maximize:
        solver.setObjective(obj, GRB.MAXIMIZE)
    else:
        solver.setObjective(obj, GRB.MINIMIZE)

    solver.update()
    solver.setParam("OutputFlag", 0)


    solver.optimize()
    if solver.status == GRB.OPTIMAL:
        # print('obj', t.X)
        x_opt = np.array([x[i].X for i in range(n_dim)])
        return x_opt
    else:
        raise ValueError("problem...")



