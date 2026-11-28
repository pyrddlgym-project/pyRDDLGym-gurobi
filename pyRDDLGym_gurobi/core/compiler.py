import math
import re
import sys
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import gurobipy
from gurobipy import GRB, nlfunc

from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer
from pyRDDLGym.core.compiler.levels import RDDLLevelAnalysis
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.compiler.tracer import RDDLObjectsTracer
from pyRDDLGym.core.debug.exception import (
    print_stack_trace,
    RDDLActionPreconditionNotSatisfiedError,
    RDDLNotImplementedError,
    RDDLStateInvariantNotSatisfiedError,
    RDDLUndefinedVariableError
)
from pyRDDLGym.core.simulator import lngamma

if TYPE_CHECKING:
    from pyRDDLGym_gurobi.core.planner import GurobiPlan

INF = GRB.INFINITY
INT = RDDLValueInitializer.INT
OP_CODE = RDDLObjectsTracer.NUMPY_OP_CODE

# numpy semantics of RDDL functions, as in the pyRDDLGym simulator (constant folding)
UNARY_NUMPY = {
    'abs': np.abs,
    'sgn': lambda x: np.sign(x).astype(INT),
    'round': lambda x: np.round(x).astype(INT),
    'floor': lambda x: np.floor(x).astype(INT),
    'ceil': lambda x: np.ceil(x).astype(INT),
    'cos': np.cos, 'sin': np.sin, 'tan': np.tan,
    'acos': np.arccos, 'asin': np.arcsin, 'atan': np.arctan,
    'cosh': np.cosh, 'sinh': np.sinh, 'tanh': np.tanh,
    'exp': np.exp, 'ln': np.log, 'sqrt': np.sqrt,
    'lngamma': lngamma,
    'gamma': lambda x: np.exp(lngamma(x))
}

BINARY_NUMPY = {
    'div': lambda x, y: np.floor_divide(x, y).astype(INT),
    'mod': lambda x, y: np.mod(x, y).astype(INT),
    'fmod': np.mod,
    'min': np.minimum,
    'max': np.maximum,
    'pow': np.power,
    'log': lambda x, y: np.log(x) / np.log(y),
    'hypot': np.hypot
}

AGGREGATION_NUMPY = {
    'sum': np.sum, 'avg': np.mean, 'prod': np.prod,
    'minimum': np.min, 'maximum': np.max,
    'forall': np.all, 'exists': np.any
}


def _not_implemented(what, expr):
    raise RDDLNotImplementedError(
        f'{what} is not supported in the Gurobi compiler.\n' +
        print_stack_trace(expr))


class GurobiRDDLCompiler:
    '''Compile a lifted RDDL domain as a Gurobi problem without grounding.'''

    GUROBI_TYPES = {'int': GRB.INTEGER, 'real': GRB.CONTINUOUS, 'bool': GRB.BINARY}

    def __init__(self, rddl: RDDLLiftedModel,
                 plan: 'GurobiPlan',
                 allow_synchronous_state: bool=True,
                 rollout_horizon: Optional[int]=None,
                 epsilon: float=1e-5,
                 model_params: Optional[Dict[str, Any]]=None,
                 verbose: int=1) -> None:
        '''Creates a new compiler for formulating a lifted RDDL domain + instance
        as a Gurobi mixed-integer non-linear optimization problem, without
        grounding. Random variables are determinized: continuous and count
        distributions by their mean, Bernoulli and Discrete by their mode.

        :param rddl: the lifted RDDL model
        :param plan: the plan or policy to optimize
        :param allow_synchronous_state: whether state-fluent can be synchronous
        :param rollout_horizon: length of the planning horizon (uses the RDDL
        defined horizon if None, which must then be finite)
        :param epsilon: small positive constant used to encode strict
        comparisons (e.g. >, ~=, sgn) and floor/ceil in Gurobi constraints
        :param model_params: dictionary of parameter name and values to
        pass to Gurobi model after compilation (merged over the defaults
        NonConvex=2, IntegralityFocus=1 and IntFeasTol=1e-9)
        :param verbose: whether to print nothing (0) or a summary (1) on compile
        '''
        if rollout_horizon is None:
            if rddl.horizon == sys.maxsize:
                raise ValueError('rollout_horizon must be set when the horizon is infinite.')
            rollout_horizon = rddl.horizon
        model_params = dict(model_params or {})
        model_params.setdefault('NonConvex', 2)
        model_params.setdefault('IntegralityFocus', 1)
        model_params.setdefault('IntFeasTol', 1e-9)
        self.rddl = rddl
        self.plan = plan
        self.horizon = rollout_horizon
        self.discount = rddl.discount
        self.epsilon = epsilon
        self.model_params = model_params
        self.verbose = int(verbose)

        # compile the initial values, levels and traced objects for the RDDL model
        self.init_values = RDDLValueInitializer(rddl).initialize()
        self.levels = RDDLLevelAnalysis(rddl, allow_synchronous_state).compute_levels()
        self.traced = RDDLObjectsTracer(rddl, cpf_levels=self.levels).trace()
        self._compile_init_state()

    def summarize_hyperparameters(self) -> None:
        '''Prints the compiler and Gurobi model hyper-parameters.'''
        print(f'Gurobi compiler hyper-params:\n'
              f'    plan              ={type(self.plan).__name__}\n'
              f'    float_equality_tol={self.epsilon}\n'
              f'    lookahead_horizon ={self.horizon}\n'
              f'    verbose           ={self.verbose}\n'
              f'Gurobi model hyper-params:\n'
              f'    user_args         ={self.model_params}')

    # ===========================================================================
    # main compilation subroutines
    # ===========================================================================

    def compile(self, init_values: Optional[Dict[str, Any]]=None,
                env: Optional[gurobipy.Env]=None):
        '''Compiles and returns the current RDDL domain as a Gurobi optimization
        problem over the rollout horizon. Also returns the action variables
        constructed at each decision epoch and the plan parameter variables.

        :param init_values: override the initial values of fluents and
        non-fluents as defined in the RDDL file (if None, then the original
        values defined in the RDDL domain + instance are used instead)
        :param env: an existing gurobi environment (if None, the default
        environment is used)
        '''
        if self.verbose >= 1:
            self.summarize_hyperparameters()
            self.plan.summarize_hyperparameters()

        # create a new Gurobi model and set the parameters
        model = gurobipy.Model(env=env) if env is not None else gurobipy.Model()
        for (name, value) in self.model_params.items():
            model.setParam(name, value)
        subs = self._compile_init_subs(init_values)
        params = self.plan.params(self, model)

        # alive: the episode has not terminated before the current step
        objective = 0
        all_action_vars = []
        alive = self._compile_alive(model, subs, self._constant(True), 'init')
        for step in range(self.horizon):
            action_vars = self.plan.actions(self, model, params, step, subs)
            all_action_vars.append(action_vars)
            subs.update(action_vars)
            if not (alive[-1] or alive[0]):
                continue
            self._compile_maxnondef_constraint(model, subs, step)
            self._compile_bool_constraints(
                model, self.rddl.preconditions, subs, step, alive, 'precond',
                RDDLActionPreconditionNotSatisfiedError)
            self._compile_bool_constraints(
                model, self.rddl.invariants, subs, step, alive, 'invariant',
                RDDLStateInvariantNotSatisfiedError)
            self._compile_cpfs(model, subs, step)
            reward = self._compile_reward(model, subs, step, alive)
            objective = objective + (self.discount ** step) * reward
            for (state, next_state) in self.rddl.next_state.items():
                subs[state] = subs[next_state]
            if step + 1 < self.horizon:
                alive = self._compile_alive(model, subs, alive, step + 1)
        model.setObjective(objective, GRB.MAXIMIZE)
        return model, all_action_vars, params

    def _compile_alive(self, model, subs, alive, step):
        '''alive and no termination condition holds in the current state.'''
        # if termination conditions, the episode is alive if it was alive in previous step
        results = [self._gurobi(expr, model, subs, step) for expr in self.rddl.terminations]
        if not results or not (alive[-1] or alive[0]):
            return alive
        elif not any(symb for (*_, symb) in results):
            return self._constant(False) if np.any([v for (v, *_) in results]) else alive

        # if any termination condition is symbolic, introduce a new variable and constraints
        values = [value if symb else np.asarray(value, np.float64)
                  for (value, _, symb) in results]
        done, *_ = self._logical_or(model, values, (), f'term{step}')
        alive_value, _, alive_symb = alive
        if not alive_symb:
            alive_value = np.asarray(alive_value, np.float64)
        return self._logical_and(model, [alive_value, 1 - done], (), f'alive{step}')

    def _compile_reward(self, model, subs, step, alive):
        # if the episode is not alive, the reward is zero
        reward, _, _ = self._gurobi(self.rddl.reward, model, subs, step)
        alive_value, _, alive_symb = alive
        if not alive_symb:
            return reward

        # otherwise, introduce a new variable and constraints to gate the reward by alive
        gated = self._new(model, GRB.CONTINUOUS, (), f'Vrewardat{step}')
        model.addConstr((alive_value == 1) >> (gated == reward), name=f'Crewardat{step}1')
        model.addConstr((alive_value == 0) >> (gated == 0), name=f'Crewardat{step}0')
        return gated

    def _compile_init_state(self) -> None:
        subs = self._compile_init_subs()
        for (name, expr, target, _) in self.traced.init_state_schedule:
            sample, _, symb = self._gurobi(expr, None, subs, 'init')
            assert not symb
            values = np.array(self.init_values[name])
            sample = np.asarray(sample).astype(values.dtype)
            if isinstance(target, np.ndarray):
                values = np.where(target, np.broadcast_to(sample, values.shape), values)
            else:
                values[target] = sample
            self.init_values[name] = values
            subs[name] = self._constant(values)

    def _compile_init_subs(self, init_values=None) -> Dict[str, Any]:
        values = dict(self.init_values)
        if init_values is not None:
            values.update(init_values)
        return {var: self._constant(value) for (var, value) in values.items()}

    def _compile_maxnondef_constraint(self, model, subs, step) -> None:
        '''At most max-nondef-actions action elements differ from their defaults.'''
        # if the number of action elements is <= to the max, no constraint is needed
        rddl = self.rddl
        total = sum(int(np.size(self.init_values[action])) 
                    for action in rddl.action_fluents)
        if rddl.max_allowed_actions >= total:
            return

        # count the number of action elements that differ from their default values
        count = 0
        for action in rddl.action_fluents:
            value, vtype, symb = subs[action]
            default = np.asarray(self.init_values[action])
            if not symb:
                count = count + int(np.sum(value != default))
            elif vtype == GRB.BINARY:
                default = default.astype(np.float64)
                count = count + (default + value - 2 * default * value).sum()
            else:
                key = f'{re.sub("[^A-z0-9]", "", action)}at{step}nondef'
                gap = None if vtype == GRB.CONTINUOUS else 1
                _, gt, lt = self._equality(model, value - default, np.shape(value), key, gap)
                count = count + (gt + lt).sum()
        model.addConstr(count <= rddl.max_allowed_actions, name=f'concurrent{step}')

    def _compile_bool_constraints(self, model, exprs, subs, step, alive, tag, error) -> None:
        '''Each expression must hold while the episode is running.'''
        alive_value, _, alive_symb = alive
        for (i, expr) in enumerate(exprs):
            value, _, symb = self._gurobi(expr, model, subs, step)
            if alive_symb:
                if not symb:
                    value = float(np.all(value))
                model.addConstr(alive_value <= value, name=f'{tag}{i}at{step}')
            elif symb:
                model.addConstr(value == 1, name=f'{tag}{i}at{step}')
            elif not np.all(value):
                raise error(
                    f'{tag} {i} can not be satisfied at decision epoch {step} by any '
                    f'choice of actions.\n' + print_stack_trace(expr))

    def _compile_cpfs(self, model, subs, step) -> None:
        rddl = self.rddl
        for cpfs in self.levels.values():
            for cpf in cpfs:
                _, expr = rddl.cpfs[cpf]
                shape = rddl.object_counts(rddl.variable_params[cpf])
                subs[cpf] = self._broadcast(self._gurobi(expr, model, subs, step), shape)

    # ===========================================================================
    # tensor helpers
    # ===========================================================================

    @staticmethod
    def _constant(value):
        value = np.asarray(value)
        if value.dtype == bool:
            vtype = GRB.BINARY
        elif np.issubdtype(value.dtype, np.integer):
            vtype = GRB.INTEGER
        else:
            vtype = GRB.CONTINUOUS
        return value, vtype, False

    @staticmethod
    def _promote_vtype(*vtypes):
        for vtype in (GRB.CONTINUOUS, GRB.INTEGER):
            if vtype in vtypes:
                return vtype
        return GRB.BINARY

    @staticmethod
    def _gather(x, index):
        if np.ndim(x) == 0:
            x = x.reshape(1)
            index = (np.zeros_like(index[0]) if index else np.zeros((), dtype=int),)
        return x[index]

    def _transpose(self, value, symb, order):
        if not symb:
            return np.transpose(value, order)
        shape = np.shape(value)
        grid = np.indices([shape[axis] for axis in order])
        index = [None] * len(order)
        for (j, axis) in enumerate(order):
            index[axis] = grid[j]
        return self._gather(value, tuple(index))

    def _moveaxis(self, value, symb, source, destination):
        # the axis permutation np.moveaxis would apply (numpy's own algorithm)
        ndim = np.ndim(value)
        source = [s % ndim for s in np.atleast_1d(source)]
        destination = [d % ndim for d in np.atleast_1d(destination)]
        order = [axis for axis in range(ndim) if axis not in source]
        for (dest, src) in sorted(zip(destination, source)):
            order.insert(dest, src)
        return self._transpose(value, symb, order)

    @staticmethod
    def _slices_along(value, axes):
        axes = tuple(np.atleast_1d(axes))
        shape = np.shape(value)
        slices = []
        for combo in np.ndindex(*(shape[axis] for axis in axes)):
            index = [slice(None)] * len(shape)
            for (axis, i) in zip(axes, combo):
                index[axis] = i
            slices.append(value[tuple(index)])
        return slices

    def _broadcast(self, result, shape):
        value, vtype, symb = result
        shape = tuple(shape)
        old_shape = np.shape(value)
        if old_shape == shape:
            return result
        elif not symb:
            return np.broadcast_to(value, shape), vtype, symb
        grid = np.indices(shape)
        keep = len(old_shape)
        index = tuple(np.where(old_shape[i] == 1, 0, grid[len(shape) - keep + i])
                      for i in range(keep))
        return self._gather(value, index), vtype, symb

    def _new(self, model, vtype, shape, name, lb=None, ub=None):
        if lb is None:
            lb, ub = (0, 1) if vtype == GRB.BINARY else (-INF, INF)
        return model.addMVar(tuple(shape), vtype=vtype, lb=lb, ub=ub, name=name)

    def _assign(self, model, expr, vtype, name):
        res = self._new(model, vtype, np.shape(expr), f'V{name}')
        model.addConstr(res == expr, name=f'C{name}')
        return res, vtype, True

    def _new_bool(self, model, shape, name):
        return self._new(model, GRB.BINARY, shape, f'V{name}')

    def _one_hot(self, model, count, shape, key):
        z = self._new_bool(model, (count,) + tuple(shape), f'{key}z')
        model.addConstr(z.sum(axis=0) == 1, name=f'C{key}z')
        return z

    @staticmethod
    def _shape_of(*values):
        return np.broadcast_shapes(*(np.shape(value) for value in values))

    # ===========================================================================
    # selection and extrema: one-hot binaries + vectorized indicators
    # ===========================================================================

    def _select(self, model, cases, index, key):
        '''res = cases[index] element-wise, for an integer tensor index.'''
        # cases: list of (value, vtype, symb) for each case
        values = [value for (value, *_) in cases]
        vtype = self._promote_vtype(*(vtype for (_, vtype, _) in cases))
        index_value, _, index_symb = index
        shape = self._shape_of(index_value, *values)

        # if the index is a constant, we can just gather the values and return a constant
        if not index_symb:
            index_value = np.broadcast_to(np.asarray(index_value, dtype=np.int64), shape)
            if not any(symb for (*_, symb) in cases):
                stacked = np.stack([np.broadcast_to(value, shape) for value in values])
                return self._constant(
                    np.take_along_axis(stacked, index_value[None], axis=0)[0])
            masks = [(index_value == k).astype(np.float64) for k in range(len(values))]
            expr = sum(mask * value for (mask, value) in zip(masks, values) if mask.any())
            return self._assign(model, expr, vtype, f'{key}sel')

        # otherwise, we need to create a one-hot selection variable and add constraints
        count = len(values)
        z = self._one_hot(model, count, shape, key)
        ks = np.arange(count).reshape((count,) + (1,) * len(shape))
        model.addConstr((ks * z).sum(axis=0) == index_value, name=f'C{key}idx')
        res = self._new(model, vtype, shape, f'V{key}sel')
        for (k, value) in enumerate(values):
            model.addConstr((z[k] == 1) >> (res == value), name=f'C{key}sel{k}')
        return res, vtype, True

    def _extremum(self, model, cases, maximize, key):
        '''res = max (min) of cases element-wise; z one-hot marks a case attaining it.'''
        values = [value for (value, *_) in cases]
        vtype = self._promote_vtype(*(vtype for (_, vtype, _) in cases))
        shape = self._shape_of(*values)
        res = self._new(model, vtype, shape, f'V{key}ext')
        z = self._one_hot(model, len(values), shape, key)
        for (k, value) in enumerate(values):
            if maximize:
                model.addConstr(res >= value, name=f'C{key}bnd{k}')
                model.addConstr((z[k] == 1) >> (res <= value), name=f'C{key}att{k}')
            else:
                model.addConstr(res <= value, name=f'C{key}bnd{k}')
                model.addConstr((z[k] == 1) >> (res >= value), name=f'C{key}att{k}')
        return res, vtype, z

    def _extremum_value(self, model, cases, maximize, key):
        if not any(symb for (*_, symb) in cases):
            op = np.maximum if maximize else np.minimum
            return self._constant(
                op.reduce(np.broadcast_arrays(*(1 * v for (v, *_) in cases))))
        res, vtype, _ = self._extremum(model, cases, maximize, key)
        return res, vtype, True

    def _argextremum(self, model, cases, maximize, key):
        '''Index of the first case attaining the max (min), as numpy.argmax (argmin).'''
        # cases: list of (value, vtype, symb) for each case
        values = [value for (value, *_) in cases]
        if not any(symb for (*_, symb) in cases):
            op = np.argmax if maximize else np.argmin
            return self._constant(op(np.stack(np.broadcast_arrays(*values)), axis=0))
        
        # first-index tie-breaking: no earlier case may attain the extremum
        best, _, z = self._extremum(model, cases, maximize, key)
        eps = self.epsilon
        for k in range(1, len(values)):
            for j in range(k):
                gap = (values[j] - best <= -eps) if maximize else (values[j] - best >= eps)
                model.addConstr((z[k] == 1) >> gap, name=f'C{key}tie{k}_{j}')
        ks = np.arange(len(values)).reshape((len(values),) + (1,) * (z.ndim - 1))
        return self._assign(model, (ks * z).sum(axis=0), GRB.INTEGER, f'{key}arg')

    def _floor(self, model, value, key, mode='floor'):
        '''Integer res = floor/ceil(value) using the epsilon convention for strictness.'''
        eps = self.epsilon
        res = self._new(model, GRB.INTEGER, np.shape(value), f'V{key}{mode}')
        if mode == 'floor':
            model.addConstr(res <= value, name=f'C{key}{mode}1')
            model.addConstr(res >= value - 1 + eps, name=f'C{key}{mode}2')
        else:
            model.addConstr(res >= value, name=f'C{key}{mode}1')
            model.addConstr(res <= value + 1 - eps, name=f'C{key}{mode}2')
        return res

    def _divmod(self, model, x, y, y_symb, rtype, key):
        '''Quotient q = floor(x / y) and remainder r = x - y * q, as numpy.'''
        # if y is a constant, we can just compute the quotient and remainder directly
        if not y_symb:
            q = self._floor(model, x * (1. / y), f'{key}q')
            r, *_ = self._assign(model, x - y * q, rtype, f'{key}r')
            return q, r

        # the remainder has the sign of y: 0 <= r < y or y < r <= 0 (y == 0 is undefined)
        eps = self.epsilon
        delta = 1 if rtype == GRB.INTEGER else eps
        shape = self._shape_of(x, y)
        y, *_ = self._broadcast((y, GRB.CONTINUOUS, True), shape)
        q = self._new(model, GRB.INTEGER, shape, f'V{key}q')
        r = self._new(model, rtype, shape, f'V{key}r')
        pos = self._new_bool(model, shape, f'{key}pos')
        model.addConstr(y * q + r == x, name=f'C{key}divmod')
        model.addConstr((pos == 1) >> (y >= eps), name=f'C{key}ypos')
        model.addConstr((pos == 0) >> (y <= -eps), name=f'C{key}yneg')
        model.addConstr((pos == 1) >> (r >= 0), name=f'C{key}rlb1')
        model.addConstr((pos == 1) >> (r - y <= -delta), name=f'C{key}rub1')
        model.addConstr((pos == 0) >> (r <= 0), name=f'C{key}rub0')
        model.addConstr((pos == 0) >> (r - y >= delta), name=f'C{key}rlb0')
        return q, r

    def _round(self, model, x, key):
        '''Integer res = round(x) with halves to even, as numpy.'''
        eps = self.epsilon
        shape = np.shape(x)
        res = self._new(model, GRB.INTEGER, shape, f'V{key}round')
        half = self._new(model, GRB.INTEGER, shape, f'V{key}half')
        odd = self._new_bool(model, shape, f'{key}odd')
        model.addConstr(res == 2 * half + odd, name=f'C{key}parity')

        # z: no tie, x - res == 0.5, x - res == -0.5; at a tie res is even
        z = self._one_hot(model, 3, shape, key)
        diff = x - res
        model.addConstr((z[0] == 1) >> (diff <= 0.5 - eps), name=f'C{key}near1')
        model.addConstr((z[0] == 1) >> (diff >= -0.5 + eps), name=f'C{key}near2')
        model.addConstr((z[1] == 1) >> (diff == 0.5), name=f'C{key}tieup')
        model.addConstr((z[2] == 1) >> (diff == -0.5), name=f'C{key}tiedown')
        model.addConstr(odd <= z[0], name=f'C{key}even')
        return res

    # ===========================================================================
    # expressions
    # ===========================================================================

    def _gurobi(self, expr, model, subs, step):
        etype, _ = expr.etype
        handler = {
            'constant': self._gurobi_constant,
            'pvar': self._gurobi_pvar,
            'arithmetic': self._gurobi_arithmetic,
            'relational': self._gurobi_relational,
            'boolean': self._gurobi_logical,
            'aggregation': self._gurobi_aggregation,
            'func': self._gurobi_function,
            'control': self._gurobi_control,
            'randomvar': self._gurobi_random,
            'randomvector': self._gurobi_random_vector,
            'matrix': self._gurobi_matrix
        }.get(etype)
        if handler is None:
            _not_implemented(f'Expression type {etype}', expr)
        return handler(expr, model, subs, step)

    def _gurobi_constant(self, expr, model, subs, step):
        return self._constant(self.traced.cached_sim_info(expr))

    def _gurobi_pvar(self, expr, model, subs, step):
        # pvar: a parameterized variable, e.g. x[i,j] or y[i] where i,j are indices
        var, pvars = expr.args
        is_value, info = self.traced.cached_sim_info(expr)
        if is_value:
            return self._constant(info)
        result = subs.get(var)
        if result is None:
            raise RDDLUndefinedVariableError(
                f'Variable <{var}> is referenced before assignment.\n' +
                print_stack_trace(expr))
        if info is None:
            return result

        # nested pvariables: evaluate the nested arguments as indices
        slices, new_axis, new_shape, op_code, op_args = info
        if op_code == OP_CODE.NESTED_SLICE:
            indices = [self._gurobi(arg, model, subs, step) if slice_ is None
                       else (slice_, GRB.INTEGER, False)
                       for (arg, slice_) in zip(pvars, slices)]
            return self._nested_gather(model, result, indices, f'{expr.id}at{step}nest')

        # one gather does the tracer's slice, expand_dims, broadcast_to,
        # transpose and einsum (generalized diagonal) all at once
        value, vtype, symb = result
        if slices:
            value = value[slices]
        new_axis = new_axis or ()
        if not new_axis and op_code == OP_CODE.NOOP:
            return value, vtype, symb
        if op_code == OP_CODE.EINSUM:
            permuted = op_args[0]
        elif op_code == OP_CODE.TRANSPOSE:
            permuted = list(np.argsort(op_args))
        else:
            permuted = list(range(len(new_shape)))
        object_shape = [0] * len(new_shape)
        for (axis, size) in zip(permuted, new_shape):
            object_shape[axis] = size
        grid = np.indices(object_shape)
        index = tuple(grid[permuted[i]] for i in range(len(new_shape) - len(new_axis)))
        return self._gather(value, index), vtype, symb

    def _nested_gather(self, model, result, indices, key):
        value, vtype, symb = result

        # an index that depends on decisions: enumerate its values and select
        for (j, (index, _, index_symb)) in enumerate(indices):
            if index_symb:
                cases = []
                for k in range(np.shape(value)[j]):
                    fixed = list(indices)
                    fixed[j] = (k, GRB.INTEGER, False)
                    cases.append(self._nested_gather(model, result, fixed, f'{key}_{j}{k}'))
                return self._select(model, cases, indices[j], f'{key}{j}')

        # all indices are constant: gather the values directly
        index = tuple(int(idx) if np.ndim(idx) == 0 else np.asarray(idx, dtype=np.int64)
                      for (idx, *_) in indices)
        return value[index], vtype, symb

    # ===========================================================================
    # arithmetic
    # ===========================================================================

    def _gurobi_arithmetic(self, expr, model, subs, step):
        _, op = expr.etype
        results = [self._gurobi(arg, model, subs, step) for arg in expr.args]
        results = [r if r[-1] else (1 * r[0],) + r[1:] for r in results]
        n = len(results)
        key = f'{expr.id}at{step}'

        if n == 1 and op == '-':
            value, vtype, symb = results[0]
            if symb:
                return self._assign(
                    model, -value, self._promote_vtype(vtype, GRB.INTEGER), f'{key}neg')
            else:
                return self._constant(-value)

        elif op in ('+', '-') and (op == '+' or n == 2):
            sign = 1 if op == '+' else -1
            value, vtype, symb = results[0]
            vtype = self._promote_vtype(vtype, GRB.INTEGER)
            for (value2, vtype2, symb2) in results[1:]:
                value = value + sign * value2
                vtype = self._promote_vtype(vtype, vtype2)
                symb = symb or symb2
            if symb:
                return self._assign(model, value, vtype, f'{key}add')
            else:
                return self._constant(value)
 
        elif op == '*':
            value, vtype, symb = results[0]
            for (i, (value2, vtype2, symb2)) in enumerate(results[1:]):
                value = value * value2
                vtype = self._promote_vtype(vtype, vtype2, GRB.INTEGER)
                symb = symb or symb2
                if symb:
                    value, vtype, symb = self._assign(model, value, vtype, f'{key}mul{i}')
            return self._constant(value) if not symb else (value, vtype, symb)

        elif op == '/' and n == 2:
            (value1, _, symb1), (value2, _, symb2) = results
            if not (symb1 or symb2):
                return self._constant(value1 / value2)
            elif not symb2:
                return self._assign(
                    model, value1 * (1. / value2), GRB.CONTINUOUS, f'{key}div')
            else:
                res = self._new(
                    model, GRB.CONTINUOUS, self._shape_of(value1, value2), f'V{key}div')
                model.addConstr(res * value2 == value1, name=f'C{key}div')
                return res, GRB.CONTINUOUS, True

        _not_implemented(f'Arithmetic operator {op} with {n} arguments', expr)

    # ===========================================================================
    # boolean
    # ===========================================================================

    def _gurobi_relational(self, expr, model, subs, step):
        _, op = expr.etype
        if len(expr.args) != 2:
            _not_implemented(
                f'Relational operator {op} with {len(expr.args)} arguments', expr)
            
        lhs, rhs = expr.args
        if op in ('<=', '<'):
            lhs, rhs, op = rhs, lhs, op.replace('<', '>')

        # if both sides are constants, we can evaluate the relational operator directly
        value1, vtype1, symb1 = self._gurobi(lhs, model, subs, step)
        value2, vtype2, symb2 = self._gurobi(rhs, model, subs, step)
        if not (symb1 or symb2):
            return self._constant({
                '>=': np.greater_equal, '>': np.greater,
                '==': np.equal, '~=': np.not_equal
            }[op](value1, value2))

        # otherwise, introduce a new variable and constraints to represent the operator:
        # integer differences are separated exactly by 1, real ones by epsilon
        diff = value1 - value2
        shape = self._shape_of(value1, value2)
        key = f'{expr.id}at{step}'
        integral = self._promote_vtype(vtype1, vtype2) != GRB.CONTINUOUS
        gap = 1 if integral else self.epsilon

        if op in ('>=', '>'):
            strict = (op == '>')
            res = self._new_bool(model, shape, f'{key}cmp')
            model.addConstr((res == 1) >> (diff >= (gap if strict else 0)),
                            name=f'C{key}cmp1')
            model.addConstr((res == 0) >> (diff <= (0 if strict else -gap)),
                            name=f'C{key}cmp0')
            return res, GRB.BINARY, True

        elif op in ('==', '~='):
            eq, gt, lt = self._equality(model, diff, shape, key, gap)
            if op == '==':
                return eq, GRB.BINARY, True
            else:
                return self._assign(model, gt + lt, GRB.BINARY, f'{key}neq')

        _not_implemented(f'Relational operator {op}', expr)

    def _equality(self, model, diff, shape, key, gap=None):
        '''Binaries eq, gt, lt: exactly one of diff == 0, diff >= gap, diff <= -gap
        (gap is epsilon by default, or 1 for integer differences).'''
        gap = self.epsilon if gap is None else gap
        eq = self._new_bool(model, shape, f'{key}eq')
        gt = self._new_bool(model, shape, f'{key}gt')
        lt = self._new_bool(model, shape, f'{key}lt')
        model.addConstr(eq + gt + lt == 1, name=f'C{key}eq')
        model.addConstr((eq == 1) >> (diff == 0), name=f'C{key}eq1')
        model.addConstr((gt == 1) >> (diff >= gap), name=f'C{key}gt1')
        model.addConstr((lt == 1) >> (diff <= -gap), name=f'C{key}lt1')
        return eq, gt, lt

    def _logical_and(self, model, values, shape, key):
        res = self._new_bool(model, shape, key)
        for (i, value) in enumerate(values):
            model.addConstr(res <= value, name=f'C{key}{i}')
        model.addConstr(res >= sum(values) - (len(values) - 1), name=f'C{key}')
        return res, GRB.BINARY, True

    def _logical_or(self, model, values, shape, key):
        res = self._new_bool(model, shape, key)
        for (i, value) in enumerate(values):
            model.addConstr(res >= value, name=f'C{key}{i}')
        model.addConstr(res <= sum(values), name=f'C{key}')
        return res, GRB.BINARY, True

    def _gurobi_logical(self, expr, model, subs, step):
        _, op = expr.etype
        op = '^' if op == '&' else op
        results = [self._gurobi(arg, model, subs, step) for arg in expr.args]
        values = [value for (value, *_) in results]
        n = len(results)
        key = f'{expr.id}at{step}'

        # if none of the arguments are symbolic, compute the logical operation directly
        if not any(symb for (*_, symb) in results):
            if n == 1 and op == '~':
                return self._constant(np.logical_not(values[0]))
            const_ops = {'^': np.logical_and.reduce, '|': np.logical_or.reduce,
                         '=>': lambda v: np.logical_or(np.logical_not(v[0]), v[1]),
                         '<=>': lambda v: np.equal(v[0], v[1])}
            if op in const_ops:
                return self._constant(const_ops[op](np.broadcast_arrays(*values)))
            _not_implemented(f'Logical operator {op} with {n} arguments', expr)

        # constants enter the linear constraints below as 0/1 arrays
        values = [value if symb else np.asarray(value, np.float64)
                  for (value, _, symb) in results]
        shape = self._shape_of(*values)
        if n == 1 and op == '~':
            return self._assign(model, 1 - values[0], GRB.BINARY, f'{key}not')
        elif op == '^':
            return self._logical_and(model, values, shape, f'{key}and')
        elif op == '|':
            return self._logical_or(model, values, shape, f'{key}or')
        elif op == '=>' and n == 2:
            return self._logical_or(model, [1 - values[0], values[1]], shape, f'{key}imply')
        elif op == '<=>' and n == 2:
            a, b = values
            res = self._new_bool(model, shape, f'{key}equiv')
            model.addConstr(res >= 1 - a - b, name=f'C{key}equiv1')
            model.addConstr(res >= a + b - 1, name=f'C{key}equiv2')
            model.addConstr(res <= 1 - a + b, name=f'C{key}equiv3')
            model.addConstr(res <= 1 + a - b, name=f'C{key}equiv4')
            return res, GRB.BINARY, True
        _not_implemented(f'Logical operator {op} with {n} arguments', expr)

    # ===========================================================================
    # aggregation
    # ===========================================================================

    def _gurobi_aggregation(self, expr, model, subs, step):
        _, op = expr.etype
        *_, arg = expr.args
        value, vtype, symb = self._gurobi(arg, model, subs, step)
        _, axes = self.traced.cached_sim_info(expr)
        axes = tuple(np.atleast_1d(axes))
        key = f'{expr.id}at{step}'

        if op in ('argmin', 'argmax'):
            if len(axes) != 1:
                _not_implemented(f'Aggregation {op} over {len(axes)} variables', expr)
            cases = [(case, vtype, symb) for case in self._slices_along(value, axes)]
            return self._argextremum(model, cases, op == 'argmax', f'{key}{op}')

        # if the value is a constant, we can compute the aggregation directly
        if not symb:
            if op not in AGGREGATION_NUMPY:
                _not_implemented(f'Aggregation {op}', expr)
            if op in ('minimum', 'maximum'):
                value = 1 * value
            return self._constant(AGGREGATION_NUMPY[op](value, axis=axes))

        # otherwise, introduce a new variable and constraints to represent the aggregation
        shape = np.shape(value)
        count = int(np.prod([shape[i] for i in axes]))
        if op in ('sum', 'avg'):
            scale = 1.0 if op == 'sum' else 1.0 / count
            vtype = self._promote_vtype(vtype, GRB.INTEGER) if op == 'sum' else GRB.CONTINUOUS
            return self._assign(model, scale * value.sum(axis=axes), vtype, f'{key}{op}')

        if op in ('forall', 'exists'):
            keep = tuple(1 if i in axes else s for (i, s) in enumerate(shape))
            res = self._new_bool(model, np.delete(shape, axes), f'{key}{op}')
            res_keep = res.reshape(keep)
            if op == 'forall':
                model.addConstr(res_keep <= value, name=f'C{key}forall1')
                model.addConstr(res >= value.sum(axis=axes) - (count - 1), 
                                name=f'C{key}forall2')
            else:
                model.addConstr(res_keep >= value, name=f'C{key}exists1')
                model.addConstr(res <= value.sum(axis=axes), name=f'C{key}exists2')
            return res, GRB.BINARY, True

        cases = [(case, vtype, True) for case in self._slices_along(value, axes)]
        if op in ('minimum', 'maximum'):
            return self._extremum_value(model, cases, op == 'maximum', f'{key}{op}')

        if op == 'prod':
            res = cases[0]
            vtype = self._promote_vtype(vtype, GRB.INTEGER)
            for (i, (case, *_)) in enumerate(cases[1:]):
                res = self._assign(model, res[0] * case, vtype, f'{key}prod{i}')
            return res

        _not_implemented(f'Aggregation {op}', expr)

    # ===========================================================================
    # functions
    # ===========================================================================

    # RDDL unary functions that map directly to nlfunc
    NLFUNC_UNARY = {'exp': 'exp', 'ln': 'log', 'sqrt': 'sqrt', 'sin': 'sin',
                    'cos': 'cos', 'tan': 'tan', 'tanh': 'tanh'}

    # inverse trig as x = f(y), with y on the principal branch (part of the definition)
    INVERSE_TRIG = {'asin': ('sin', -math.pi / 2, math.pi / 2),
                    'acos': ('cos', 0.0, math.pi),
                    'atan': ('tan', -math.pi / 2 + 1e-9, math.pi / 2 - 1e-9)}

    def _gurobi_function(self, expr, model, subs, step):
        _, name = expr.etype
        results = [self._gurobi(arg, model, subs, step) for arg in expr.args]
        n = len(results)
        key = f'{expr.id}at{step}{name}'
        if n == 1 and name in UNARY_NUMPY or n == 2 and name in BINARY_NUMPY:
            if not any(symb for (*_, symb) in results):
                op = UNARY_NUMPY.get(name) if n == 1 else BINARY_NUMPY.get(name)
                return self._constant(op(*(1 * value for (value, *_) in results)))
            elif n == 1:
                return self._gurobi_unary(name, results[0], model, key, expr)
            else:
                return self._gurobi_binary(name, results, model, key, expr)
        _not_implemented(f'Function {name} with {n} arguments', expr)

    def _gurobi_unary(self, name, result, model, key, expr):
        x, vtype, _ = result
        eps = self.epsilon

        if name in self.NLFUNC_UNARY:
            return self._assign(
                model, getattr(nlfunc, self.NLFUNC_UNARY[name])(x), GRB.CONTINUOUS, key)
        
        elif name in ('sinh', 'cosh'):
            sign = -1 if name == 'sinh' else 1
            return self._assign(
                model, (nlfunc.exp(x) + sign * nlfunc.exp(-x)) * 0.5, GRB.CONTINUOUS, key)
        
        elif name in self.INVERSE_TRIG:
            forward, lb, ub = self.INVERSE_TRIG[name]
            res = self._new(model, GRB.CONTINUOUS, np.shape(x), f'V{key}', lb=lb, ub=ub)
            model.addConstr(x == getattr(nlfunc, forward)(res), name=f'C{key}')
            return res, GRB.CONTINUOUS, True
        
        elif name == 'abs':
            vtype = self._promote_vtype(vtype, GRB.INTEGER)
            return self._extremum_value(
                model, [(x, vtype, True), (-x, vtype, True)], True, key)
        
        elif name == 'sgn':
            pos = self._new_bool(model, np.shape(x), f'{key}pos')
            neg = self._new_bool(model, np.shape(x), f'{key}neg')
            model.addConstr((pos == 1) >> (x >= eps), name=f'C{key}pos1')
            model.addConstr((pos == 0) >> (x <= 0), name=f'C{key}pos0')
            model.addConstr((neg == 1) >> (x <= -eps), name=f'C{key}neg1')
            model.addConstr((neg == 0) >> (x >= 0), name=f'C{key}neg0')
            return self._assign(model, pos - neg, GRB.INTEGER, key)
        
        elif name in ('floor', 'ceil'):
            if vtype != GRB.CONTINUOUS:  # already integral
                return x, GRB.INTEGER, True
            return self._floor(model, x, key, name), GRB.INTEGER, True
        
        elif name == 'round':
            return self._round(model, x, key), GRB.INTEGER, True
        
        _not_implemented(f'Function {name} over decision variables', expr)

    def _gurobi_binary(self, name, results, model, key, expr):
        (x, vtype1, symb1), (y, vtype2, symb2) = results

        if name in ('min', 'max'):
            return self._extremum_value(model, results, name == 'max', key)
        
        elif name in ('div', 'mod', 'fmod'):
            rtype = GRB.CONTINUOUS if name == 'fmod' else GRB.INTEGER
            q, r = self._divmod(model, 1 * x if not symb1 else x, y, symb2, rtype, key)
            return (q, GRB.INTEGER, True) if name == 'div' else (r, rtype, True)
        
        elif name == 'pow':
            return self._pow(model, x, symb1, y, symb2, key)
        
        elif name == 'log':
            log_x = nlfunc.log(x) if symb1 else np.log(x)
            if symb2:
                return self._assign(model, log_x / nlfunc.log(y), GRB.CONTINUOUS, key)
            else:
                return self._assign(model, log_x * (1. / np.log(y)), GRB.CONTINUOUS, key)
            
        elif name == 'hypot':
            return self._assign(model, nlfunc.sqrt(x * x + y * y), GRB.CONTINUOUS, key)

        _not_implemented(f'Function {name} over decision variables', expr)

    def _pow(self, model, x, symb1, y, symb2, key):
        '''x ** y wherever numpy defines it as a finite real number.'''

        # constant exponent: Gurobi's POW is valid for integer powers of negative bases
        if not symb2:
            y = np.asarray(y, dtype=np.float64)
            if np.all(y == y.flat[0]):
                power = x ** float(y.flat[0])
            else:
                power = x ** np.broadcast_to(y, self._shape_of(x, y))
            return self._assign(model, power, GRB.CONTINUOUS, key)

        # constant positive base
        if not symb1:
            x = np.asarray(x, dtype=np.float64)
            if np.all(x > 0):
                return self._assign(model, x ** y, GRB.CONTINUOUS, key)
            x, *_ = self._assign(model, x, GRB.CONTINUOUS, f'{key}base')

        # z: base > 0, base < 0, base == 0
        eps = self.epsilon
        shape = self._shape_of(x, y)
        x, *_ = self._broadcast((x, GRB.CONTINUOUS, True), shape)
        y, *_ = self._broadcast((y, GRB.CONTINUOUS, True), shape)
        z = self._one_hot(model, 3, shape, f'{key}sign')
        model.addConstr((z[0] == 1) >> (x >= eps), name=f'C{key}xpos')
        model.addConstr((z[1] == 1) >> (x <= -eps), name=f'C{key}xneg')
        model.addConstr((z[2] == 1) >> (x == 0), name=f'C{key}xzero')

        # magnitude |x| ** y = exp(y log |x|), with a dummy base of 1 when x == 0
        base = self._new(model, GRB.CONTINUOUS, shape, f'V{key}abs')
        model.addConstr((z[0] == 1) >> (base == x), name=f'C{key}abs0')
        model.addConstr((z[1] == 1) >> (base == -x), name=f'C{key}abs1')
        model.addConstr((z[2] == 1) >> (base == 1), name=f'C{key}abs2')
        magnitude, *_ = self._assign(
            model, nlfunc.exp(y * nlfunc.log(base)), GRB.CONTINUOUS, f'{key}mag')

        # negative base: only integer exponents, with sign (-1) ** y
        half = self._new(model, GRB.INTEGER, shape, f'V{key}half')
        parity = self._new_bool(model, shape, f'{key}parity')
        model.addConstr((z[1] == 1) >> (y - 2 * half - parity == 0), name=f'C{key}int')
        flip, *_ = self._logical_and(model, [z[1], parity], shape, f'{key}flip')
        flipped = self._new(model, GRB.CONTINUOUS, shape, f'V{key}flipped')
        model.addConstr((flip == 1) >> (flipped == magnitude), name=f'C{key}flip1')
        model.addConstr((flip == 0) >> (flipped == 0), name=f'C{key}flip0')

        # zero base: 0 ** 0 == 1, 0 ** y == 0 for y > 0, and y < 0 is undefined
        one = self._new_bool(model, shape, f'{key}zero0')
        zero = self._new_bool(model, shape, f'{key}zero1')
        model.addConstr(one + zero == z[2], name=f'C{key}zero')
        model.addConstr((one == 1) >> (y == 0), name=f'C{key}zeroexp0')
        model.addConstr((zero == 1) >> (y >= eps), name=f'C{key}zeroexp1')

        # the result is magnitude with the sign of flipped, or 1 when x == 0 and y == 0
        res = self._new(model, GRB.CONTINUOUS, shape, f'V{key}')
        model.addConstr((z[2] == 0) >> (res - magnitude + 2 * flipped == 0), name=f'C{key}1')
        model.addConstr((z[2] == 1) >> (res - one == 0), name=f'C{key}0')
        return res, GRB.CONTINUOUS, True

    # ===========================================================================
    # control
    # ===========================================================================

    def _gurobi_control(self, expr, model, subs, step):
        _, op = expr.etype
        if op == 'switch':
            return self._gurobi_switch(expr, model, subs, step)
        elif op != 'if':
            _not_implemented(f'Control flow {op}', expr)

        pred, arg1, arg2 = expr.args
        valuep, _, symbp = self._gurobi(pred, model, subs, step)

        # constant predicate: prune dead branches, otherwise blend linearly
        if not symbp:
            if np.all(valuep):
                return self._gurobi(arg1, model, subs, step)
            elif not np.any(valuep):
                return self._gurobi(arg2, model, subs, step)

        # if the predicate is symbolic, compile both branches and select the result
        value1, vtype1, symb1 = self._gurobi(arg1, model, subs, step)
        value2, vtype2, symb2 = self._gurobi(arg2, model, subs, step)
        vtype = self._promote_vtype(vtype1, vtype2)
        key = f'{expr.id}at{step}if'
        if not symbp:
            if not (symb1 or symb2):
                return self._constant(np.where(valuep, value1, value2))
            p = np.asarray(valuep, np.float64)
            return self._assign(model, p * value1 + (1 - p) * value2, vtype, key)

        # if the predicate is symbolic, introduce a new variable and constraints
        shape = self._shape_of(valuep, value1, value2)
        res = self._new(model, vtype, shape, f'V{key}')
        valuep, *_ = self._broadcast((valuep, GRB.BINARY, True), shape)
        model.addConstr((valuep == 1) >> (res == value1), name=f'C{key}1')
        model.addConstr((valuep == 0) >> (res == value2), name=f'C{key}0')
        return res, vtype, True

    def _gurobi_switch(self, expr, model, subs, step):
        pred, *_ = expr.args
        index = self._gurobi(pred, model, subs, step)
        cases, default = self.traced.cached_sim_info(expr)

        # constant predicate: only the selected case is compiled
        index_value, _, index_symb = index
        if not index_symb:
            first = int(np.ravel(index_value)[0])
            if np.all(np.asarray(index_value) == first):
                arg = default if cases[first] is None else cases[first]
                return self._gurobi(arg, model, subs, step)

        # if the predicate is symbolic, compile all cases and select the result
        default_result = None
        results = []
        for arg in cases:
            if arg is None:
                if default_result is None:
                    default_result = self._gurobi(default, model, subs, step)
                results.append(default_result)
            else:
                results.append(self._gurobi(arg, model, subs, step))
        return self._select(model, results, index, f'{expr.id}at{step}switch')

    # ===========================================================================
    # random variables (determinized: mean, or mode for Bernoulli/Discrete)
    # ===========================================================================

    MEANS = {
        'Normal': lambda mean, var: mean,
        'Uniform': lambda a, b: (a + b) * 0.5,
        'Poisson': lambda rate: rate,
        'Exponential': lambda scale: scale,
        'Gamma': lambda shape, scale: shape * scale,
        'Binomial': lambda n, p: n * p,
        'NegativeBinomial': lambda n, p: n * (1 - p) / p,
        'Beta': lambda a, b: a / (a + b),
        'Geometric': lambda p: 1.0 / p,
        'Pareto': lambda shape, scale: scale * shape / (shape - 1),
        'Student': lambda df: 0 * df,
        'Gumbel': lambda mean, scale: mean + np.euler_gamma * scale,
        'Laplace': lambda mean, scale: mean,
        'Cauchy': lambda mean, scale: mean,  # mean undefined: use the median
        'ChiSquare': lambda df: df,
    }

    @staticmethod
    def _constant_mean(name, *args):
        '''Means that need special functions: parameters must be constants.'''
        if name == 'Kumaraswamy':
            a, b = args
            return b * np.exp(lngamma(1. + 1. / a) + lngamma(b) - lngamma(1. + 1. / a + b))
        elif name == 'Gompertz':
            from scipy.special import exp1
            shape, scale = args
            return np.exp(shape) * exp1(shape) / scale
        else:
            return None

    def _gurobi_random(self, expr, model, subs, step):
        _, name = expr.etype
        key = f'{expr.id}at{step}{name}'

        if name in ('KronDelta', 'DiracDelta'):
            arg, = expr.args
            return self._gurobi(arg, model, subs, step)

        elif name == 'Bernoulli':
            arg, = expr.args
            value, _, symb = self._gurobi(arg, model, subs, step)
            if not symb:
                return self._constant(np.greater(value, 0.5))
            res = self._new_bool(model, np.shape(value), key)
            model.addConstr((res == 1) >> (value >= 0.5 + self.epsilon), name=f'C{key}1')
            model.addConstr((res == 0) >> (value <= 0.5), name=f'C{key}0')
            return res, GRB.BINARY, True

        elif name in ('Discrete', 'UnnormDiscrete', 'Discrete(p)', 'UnnormDiscrete(p)'):
            if name.endswith('(p)'):
                _, (arg,) = expr.args
                value, vtype, symb = self._gurobi(arg, model, subs, step)
                cases = [(case, vtype, symb) for case in self._slices_along(value, -1)]
            else:
                cases = [self._gurobi(arg, model, subs, step)
                         for arg in self.traced.cached_sim_info(expr)]
            return self._argextremum(model, cases, True, key)  # the mode

        results = [self._gurobi(arg, model, subs, step) for arg in expr.args]
        values = [1 * value if not symb else value for (value, _, symb) in results]
        symb = any(symb for (*_, symb) in results)
        if name == 'Weibull' and not results[0][-1]:
            shape, scale = values
            mean = scale * np.exp(lngamma(1. + 1. / np.asarray(shape, dtype=np.float64)))
            if symb:
                return self._assign(model, mean, GRB.CONTINUOUS, key)
            else:
                return self._constant(np.asarray(mean, dtype=np.float64))

        # if the distribution has a known mean, compute it directly
        elif name in self.MEANS:
            mean = self.MEANS[name](*values)
            if symb:
                return self._assign(model, mean, GRB.CONTINUOUS, key)
            else:
                return self._constant(np.asarray(mean, dtype=np.float64))

        # if none of the arguments are symbolic, compute the mean directly
        if not symb:
            mean = self._constant_mean(name, *values)
            if mean is not None:
                return self._constant(np.asarray(mean, dtype=np.float64))
            
        _not_implemented(
            f'Distribution {name}' 
            + (' with decision-dependent parameters' if symb else ''), expr)

    def _gurobi_random_vector(self, expr, model, subs, step):
        _, name = expr.etype
        _, args = expr.args
        results = [self._gurobi(arg, model, subs, step) for arg in args]
        index, = self.traced.cached_sim_info(expr)
        key = f'{expr.id}at{step}{name}'

        if name in ('MultivariateNormal', 'MultivariateStudent'):
            value, _, symb = results[0]

        elif name == 'Dirichlet':
            alpha, _, symb = results[0]
            if not symb:
                value = alpha / np.sum(alpha, axis=-1, keepdims=True)
            else:
                total, *_ = self._assign(
                    model, alpha.sum(axis=-1), GRB.CONTINUOUS, f'{key}tot')
                total = total.reshape(np.shape(alpha)[:-1] + (1,))
                value, *_ = self._assign(model, alpha / total, GRB.CONTINUOUS, key)

        elif name == 'Multinomial':
            (trials, _, symb1), (prob, _, symb2) = results
            symb = symb1 or symb2
            trials = trials.reshape(np.shape(trials) + (1,)) if symb1 \
                else np.expand_dims(trials, -1)
            value = trials * prob
            if symb:
                value, *_ = self._assign(model, value, GRB.CONTINUOUS, key)

        else:
            _not_implemented(f'Multivariate distribution {name}', expr)

        # the vector is sampled along the last axis: move it as the simulator does
        if not symb:
            value = np.asarray(value, dtype=np.float64)
        return self._moveaxis(value, symb, -1, index), GRB.CONTINUOUS, symb

    # ===========================================================================
    # matrix algebra (constant arguments only)
    # ===========================================================================

    def _gurobi_matrix(self, expr, model, subs, step):
        _, op = expr.etype
        *_, arg = expr.args
        value, _, symb = self._gurobi(arg, model, subs, step)
        key = f'{expr.id}at{step}{op}'
        if op not in ('det', 'inverse', 'pinverse', 'cholesky'):
            _not_implemented(f'Matrix operation {op}', expr)

        # if none of the arguments are symbolic, compute the matrix operation directly
        if not symb:
            if op == 'det':
                return self._constant(np.linalg.det(value))
            ops = {'inverse': np.linalg.inv, 'pinverse': np.linalg.pinv,
                   'cholesky': np.linalg.cholesky}
            result = ops[op](value)

        elif op == 'det':
            return self._det(model, value, key), GRB.CONTINUOUS, True
        elif op == 'inverse':
            result = self._inverse(model, value, key)
        elif op == 'pinverse':
            result = self._pinverse(model, value, key)
        else:
            result = self._cholesky(model, value, key)

        # matrix dimensions are the last two axes: move them as the simulator does
        indices = self.traced.cached_sim_info(expr)
        if not symb:
            return self._constant(np.moveaxis(result, source=(-2, -1), destination=indices))
        return self._moveaxis(result, True, (-2, -1), indices), GRB.CONTINUOUS, True

    # matrices are tensors whose last two axes are rows and columns (others are batch)
    @staticmethod
    def _entry(matrix, i, j):
        return matrix[(slice(None),) * (np.ndim(matrix) - 2) + (i, j)]

    def _matmul_entry(self, a, b, i, k):
        return sum(self._entry(a, i, j) * self._entry(b, j, k) 
                   for j in range(np.shape(a)[-1]))

    def _det(self, model, a, key):
        '''Determinant by cofactor expansion, memoizing minors to keep it quadratic.'''
        n = np.shape(a)[-1]
        minors = {}
  
        # determinant of the last len(cols) rows on columns cols
        def minor(cols):
            row = n - len(cols)
            if len(cols) == 1:
                return self._entry(a, row, cols[0])
            if cols not in minors:
                expr = 0
                for (pos, j) in enumerate(cols):
                    sub = minor(cols[:pos] + cols[pos + 1:])
                    expr = expr + (-1) ** pos * (self._entry(a, row, j) * sub)
                minors[cols], *_ = self._assign(
                    model, expr, GRB.CONTINUOUS, f'{key}{len(minors)}')
            return minors[cols]
        return minor(tuple(range(n)))

    def _inverse(self, model, a, key):
        '''X with A X = I.'''
        n = np.shape(a)[-1]
        x = self._new(model, GRB.CONTINUOUS, np.shape(a), f'V{key}')
        for i in range(n):
            for k in range(n):
                model.addConstr(self._matmul_entry(a, x, i, k) == float(i == k),
                                name=f'C{key}{i}_{k}')
        return x

    def _pinverse(self, model, a, key):
        '''Moore-Penrose X: P = A X, Q = X A symmetric, P A = A and X P = X.'''
        *batch, m, n = np.shape(a)
        x = self._new(model, GRB.CONTINUOUS, (*batch, n, m), f'V{key}')
        p = self._new(model, GRB.CONTINUOUS, (*batch, m, m), f'V{key}P')
        q = self._new(model, GRB.CONTINUOUS, (*batch, n, n), f'V{key}Q')
        for (prod, left, right, size) in [(p, a, x, m), (q, x, a, n)]:
            for i in range(size):
                for k in range(size):
                    model.addConstr(
                        self._entry(prod, i, k) == self._matmul_entry(left, right, i, k))
                    if i < k:
                        model.addConstr(self._entry(prod, i, k) == self._entry(prod, k, i))
        for i in range(m):
            for k in range(n):
                model.addConstr(self._matmul_entry(p, a, i, k) == self._entry(a, i, k))
        for i in range(n):
            for k in range(m):
                model.addConstr(self._matmul_entry(x, p, i, k) == self._entry(x, i, k))
        return x

    def _cholesky(self, model, a, key):
        '''Lower-triangular L with non-negative diagonal and L L' = A (lower part of A).'''
        n = np.shape(a)[-1]
        rows, cols = np.indices((n, n))
        lb = np.broadcast_to(np.where(rows > cols, -INF, 0.), np.shape(a))
        ub = np.broadcast_to(np.where(rows >= cols, INF, 0.), np.shape(a))
        chol = self._new(model, GRB.CONTINUOUS, np.shape(a), f'V{key}', lb=lb, ub=ub)
        for i in range(n):
            for j in range(i + 1):
                model.addConstr(
                    sum(self._entry(chol, i, k) * self._entry(chol, j, k) 
                        for k in range(j + 1))
                    == self._entry(a, i, j), name=f'C{key}{i}_{j}')
        return chol
