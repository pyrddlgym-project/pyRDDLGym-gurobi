from ast import literal_eval
import configparser
import os
import re
import sys
from typing import Any, Dict, List, Tuple, Optional

Kwargs = Dict[str, Any]

import numpy as np
import gurobipy
from gurobipy import GRB

from pyRDDLGym.core.debug.exception import raise_warning, RDDLNotImplementedError
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.policy import BaseAgent

from pyRDDLGym_gurobi.core.compiler import GurobiRDDLCompiler

UNBOUNDED = (-GRB.INFINITY, +GRB.INFINITY)


def _value_of(param) -> np.ndarray:
    '''The value of a (value, vtype, symb) plan parameter, solved if symbolic.'''
    value, _, symb = param
    return np.asarray(value.X if symb else value)

# ***********************************************************************
# CONFIG FILE MANAGEMENT
# 
# - read config files from file path
# - extract experiment settings
# - instantiate planner
#
# ***********************************************************************


def _parse_config_file(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'File {path} does not exist.')
    config = configparser.RawConfigParser()
    config.optionxform = str 
    config.read(path)
    args = {k: literal_eval(v) 
            for section in config.sections()
            for (k, v) in config.items(section)}
    return config, args


def _parse_config_string(value: str):
    config = configparser.RawConfigParser()
    config.optionxform = str 
    config.read_string(value)
    args = {k: literal_eval(v) 
            for section in config.sections()
            for (k, v) in config.items(section)}
    return config, args


def _getattr_any(packages, item):
    for package in packages:
        loaded = getattr(package, item, None)
        if loaded is not None:
            return loaded
    return None


def _load_config(config, args):
    gurobi_args = {k: args[k] for (k, _) in config.items('Gurobi')}
    compiler_args = {k: args[k] for (k, _) in config.items('Optimizer')}
    
    # policy class
    plan_method = compiler_args.pop('method')
    plan_kwargs = compiler_args.pop('method_kwargs', {})
    compiler_args['plan'] = getattr(sys.modules[__name__], plan_method)(**plan_kwargs)
    compiler_args['model_params'] = gurobi_args
    
    return compiler_args


def load_config(path: str) -> Kwargs:
    '''Loads a config file at the specified file path.'''
    config, args = _parse_config_file(path)
    return _load_config(config, args)


def load_config_from_string(value: str) -> Kwargs:
    '''Loads config file contents specified explicitly as a string value.'''
    config, args = _parse_config_string(value)
    return _load_config(config, args)


# ***********************************************************************
# ALL VERSIONS OF GUROBI PLANS
#
# - straight line plan
# - piecewise linear policy
#
# ***********************************************************************


class GurobiPlan:
    '''Base class for all Gurobi compiled policies or plans.'''

    def __init__(self, action_bounds: Optional[Dict[str, Tuple[Any, Any]]]=None) -> None:
        '''Creates a new plan or policy.

        :param action_bounds: optional dictionary mapping non-bool, non-enum
        action-fluents to (lower, upper) bounds: keyed by the lifted name with
        bounds that are scalars or tensors broadcastable to the action's shape,
        or by a grounded name (e.g. release___t1) for a single element; actions
        are unbounded otherwise, since action-preconditions are compiled as constraints
        '''
        if action_bounds is None:
            action_bounds = {}
        self.action_bounds = action_bounds

    def _bounds(self, compiled: GurobiRDDLCompiler, action: str) -> Tuple[np.ndarray, np.ndarray]:
        '''Element-wise bounds on an action-fluent.'''
        rddl = compiled.rddl
        prange = rddl.action_ranges[action]
        shape = np.shape(compiled.init_values[action])
        if prange == 'bool':
            bounds = (0, 1)
        elif prange in rddl.type_to_objects:
            bounds = (0, len(rddl.type_to_objects[prange]) - 1)
        else:
            bounds = self.action_bounds.get(action, UNBOUNDED)
        lb, ub = (np.array(np.broadcast_to(np.asarray(b, dtype=np.float64), shape))
                  for b in bounds)
        if prange in ('int', 'real'):
            for (i, name) in enumerate(rddl.variable_groundings[action]):
                if name != action and name in self.action_bounds:
                    lb.flat[i], ub.flat[i] = self.action_bounds[name]
        return np.clip(lb, *UNBOUNDED), np.clip(ub, *UNBOUNDED)

    def validate(self, compiled: GurobiRDDLCompiler) -> None:
        '''Warns about action_bounds keys that name no int or real action-fluent.'''
        rddl = compiled.rddl
        known = set()
        for (action, prange) in rddl.action_ranges.items():
            if prange in ('int', 'real'):
                known.add(action)
                known.update(rddl.variable_groundings[action])
        unknown = sorted(set(self.action_bounds) - known)
        if unknown:
            raise_warning(f'action_bounds ignored for {unknown}: not int or real '
                          f'action-fluents (or their groundings).', 'red')

    def summarize_hyperparameters(self) -> None:
        pass

    def params(self, compiled: GurobiRDDLCompiler,
               model: gurobipy.Model,
               values: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        '''Returns the parameters of this plan/policy to be optimized.

        :param compiled: A gurobi compiler where the current plan is initialized
        :param model: the gurobi model instance
        :param values: if not None, freeze policy parameters to these values
        '''
        raise NotImplementedError

    def init_params(self, compiled: GurobiRDDLCompiler,
                    model: gurobipy.Model) -> Dict[str, Any]:
        '''Return initial parameter values for the current policy class, used to
        warm-start Gurobi (none by default).

        :param compiled: A gurobi compiler where the current plan is initialized
        :param model: the gurobi model instance
        '''
        return {}

    def warm_start(self, compiled: GurobiRDDLCompiler,
                   params: Dict[str, Any]) -> Dict[str, Any]:
        '''Parameter values to warm-start the problem of the next decision epoch
        when replanning, from the solved parameters of the current one.

        :param compiled: A gurobi compiler where the current plan is initialized
        :param params: solved parameter variables of the plan/policy
        '''
        return {name: _value_of(param) for (name, param) in params.items()}

    def actions(self, compiled: GurobiRDDLCompiler,
                model: gurobipy.Model,
                params: Dict[str, Any],
                step: int,
                subs: Dict[str, Any]) -> Dict[str, Any]:
        '''Returns a dictionary of action variables predicted by the plan.

        :param compiled: A gurobi compiler where the current plan is initialized
        :param model: the gurobi model instance
        :param params: parameter variables of the plan/policy
        :param step: the decision epoch
        :param subs: the set of fluent and non-fluent variables available at the
        current step
        '''
        raise NotImplementedError

    def evaluate(self, compiled: GurobiRDDLCompiler,
                 params: Dict[str, Any],
                 step: int,
                 subs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        '''Evaluates the current policy with state variables in subs, returning
        each action-fluent as a tensor.

        :param compiled: A gurobi compiler where the current plan is initialized
        :param params: parameter variables of the plan/policy
        :param step: the decision epoch
        :param subs: the fluent and non-fluent tensors at the current step
        '''
        raise NotImplementedError

    def to_string(self, compiled: GurobiRDDLCompiler,
                  params: Dict[str, Any]) -> str:
        '''Returns a string representation of the current policy.

        :param params: parameter variables of the plan/policy
        :param compiled: A gurobi compiler where the current plan is initialized
        '''
        raise NotImplementedError


class GurobiStraightLinePlan(GurobiPlan):
    '''A straight-line open-loop plan in Gurobi, with one tensor of variables
    per action-fluent and decision epoch.'''

    def params(self, compiled: GurobiRDDLCompiler,
               model: gurobipy.Model,
               values: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        rddl = compiled.rddl
        params = {}
        for (action, prange) in rddl.action_ranges.items():
            vtype = compiled.GUROBI_TYPES.get(prange, GRB.INTEGER)
            lb, ub = self._bounds(compiled, action)
            ascii_name = re.sub('[^A-z0-9 -]', '', action).replace(' ', '')
            for step in range(compiled.horizon):
                name = f'{action}__{step}'
                if values is None:
                    var = model.addMVar(lb.shape, vtype=vtype, lb=lb, ub=ub,
                                        name=f'{ascii_name}at{step}')
                    params[name] = (var, vtype, True)
                else:
                    params[name] = (np.asarray(values[name]), vtype, False)
        return params

    def init_params(self, compiled: GurobiRDDLCompiler, model: gurobipy.Model) -> Dict[str, Any]:
        return {f'{action}__{step}': compiled.init_values[action]
                for action in compiled.rddl.action_fluents
                for step in range(compiled.horizon)}

    def warm_start(self, compiled: GurobiRDDLCompiler, 
                   params: Dict[str, Any]) -> Dict[str, Any]:
        start = self.init_params(compiled, None)
        for action in compiled.rddl.action_fluents:
            for step in range(compiled.horizon - 1):
                start[f'{action}__{step}'] = _value_of(params[f'{action}__{step + 1}'])
        return start

    def actions(self, compiled: GurobiRDDLCompiler,
                model: gurobipy.Model,
                params: Dict[str, Any],
                step: int,
                subs: Dict[str, Any]) -> Dict[str, Any]:
        return {action: params[f'{action}__{step}'] 
                for action in compiled.rddl.action_fluents}

    def evaluate(self, compiled: GurobiRDDLCompiler,
                 params: Dict[str, Any],
                 step: int,
                 subs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        rddl = compiled.rddl

        # past the planning horizon, return no-op actions
        if step >= compiled.horizon:
            if step == compiled.horizon:
                raise_warning(f'Decision epoch {step} is past the planning horizon '
                              f'{compiled.horizon}: using no-op actions.', 'red')
            return {action: np.asarray(compiled.init_values[action])
                    for action in rddl.action_fluents}

        # otherwise, return the solved action values clipped to their bounds
        values = {}
        for (action, prange) in rddl.action_ranges.items():
            value = _value_of(params[f'{action}__{step}'])
            value = np.clip(value, *self._bounds(compiled, action))
            if prange == 'bool':
                value = value > 0.5
            elif prange != 'real':
                value = np.round(value).astype(np.int64)
            values[action] = value
        return values

    def to_string(self, compiled: GurobiRDDLCompiler, params: Dict[str, Any]) -> str:
        res = ''
        for step in range(compiled.horizon):
            values = self.evaluate(compiled, params, step, {})
            res += ', '.join(f'{action}[{step}] = {value}'
                             for (action, value) in values.items()) + '\n'
        return res


class GurobiPiecewisePolicy(GurobiPlan):
    '''A piecewise linear policy in Gurobi. Each action-fluent tensor has
    num_cases cases and an else case: case k holds element-wise when
    lb_k <= c_k(s) <= ub_k, at most one case holds, and the action is the value
    of the case that holds (the else case if none does).
    '''

    def __init__(self, *args,
                 state_bounds: Optional[Dict[str, Tuple[Any, Any]]]=None,
                 dependencies_constr: Optional[Dict[str, List[str]]]=None,
                 dependencies_values: Optional[Dict[str, List[str]]]=None,
                 num_cases: int=1,
                 **kwargs) -> None:
        '''Creates a new piecewise linear policy.

        :param action_bounds: as in GurobiPlan
        :param state_bounds: optional (lower, upper) bounds of state-fluents,
        used to bound and initialize the case bounds when c_k(s) is a state-fluent
        :param dependencies_constr: action-fluent -> state-fluents in c_k(s)
        :param dependencies_values: action-fluent -> state-fluents in case values
        :param num_cases: number of cases (excluding the else case)
        '''
        super(GurobiPiecewisePolicy, self).__init__(*args, **kwargs)
        self.state_bounds = state_bounds or {}
        self.dependencies_constr = dependencies_constr or {}
        self.dependencies_values = dependencies_values or {}
        self.num_cases = num_cases

    def summarize_hyperparameters(self) -> None:
        print(f'Gurobi policy hyper-params:\n'
              f'    num_cases     ={self.num_cases}\n'
              f'    state_bounds  ={self.state_bounds}\n'
              f'    constraint_dep={self.dependencies_constr}\n'
              f'    value_dep     ={self.dependencies_values}')

    def _cases(self):
        return list(range(self.num_cases)) + ['else']

    def _constr_states(self, compiled, action):
        return self.dependencies_constr.get(action, list(compiled.rddl.state_fluents))

    def _is_identity(self, compiled, action):
        '''c_k(s) is a single state-fluent with the action's shape, or a scalar.'''
        states = self._constr_states(compiled, action)
        return len(states) == 1 and np.shape(compiled.init_values[states[0]]) in (
            np.shape(compiled.init_values[action]), ())

    def _state_bounds(self, compiled, action):
        shape = np.shape(compiled.init_values[action])
        state, = self._constr_states(compiled, action)
        return tuple(np.broadcast_to(np.asarray(b, dtype=np.float64), shape)
                     for b in self.state_bounds.get(state, UNBOUNDED))

    @staticmethod
    def _contract(weight, state, action_ndim):
        '''sum_j weight[..., j] * state[j]: weight has the action's axes, then the state's.'''
        if np.ndim(state) == 0:
            return weight * state
        state = state.reshape((1,) * action_ndim + np.shape(state))
        return (weight * state).sum(axis=tuple(range(action_ndim, np.ndim(weight))))

    def _param_shapes(self, compiled, action):
        '''Shapes of the parameters of an action-fluent, keyed by parameter name.'''
        shape = np.shape(compiled.init_values[action])
        shapes = {}
        for k in self._cases():
            if k != 'else':
                if not self._is_identity(compiled, action):
                    for state in self._constr_states(compiled, action):
                        shapes[f'constr_weight__{action}__{k}__{state}'] = \
                            shape + np.shape(compiled.init_values[state])
                shapes[f'lb__{action}__{k}'] = shape
                shapes[f'ub__{action}__{k}'] = shape
            shapes[f'value_weight__{action}__{k}__bias'] = shape
            for state in self.dependencies_values.get(action, []):
                shapes[f'value_weight__{action}__{k}__{state}'] = \
                    shape + np.shape(compiled.init_values[state])
        return shapes

    def params(self, compiled: GurobiRDDLCompiler,
               model: gurobipy.Model,
               values: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        rddl = compiled.rddl
        params = {}
        for (action, prange) in rddl.action_ranges.items():
            atype = compiled.GUROBI_TYPES.get(prange, GRB.INTEGER)
            identity = self._is_identity(compiled, action)
            bias_only = not self.dependencies_values.get(action)
            for (name, shape) in self._param_shapes(compiled, action).items():

                # a constant case value has the action's type and bounds
                vtype, lb, ub = GRB.CONTINUOUS, -GRB.INFINITY, GRB.INFINITY
                if name.startswith('value_weight__') and bias_only:
                    vtype = atype
                    lb, ub = self._bounds(compiled, action)
                elif (name.startswith('lb__') or name.startswith('ub__')) and identity:
                    lb, ub = self._state_bounds(compiled, action)
                    lb, ub = lb - 1, ub + 1
                if values is not None:
                    params[name] = (np.asarray(values[name]), vtype, False)
                else:
                    ascii_name = re.sub('[^A-z0-9_]', '', name)
                    var = model.addMVar(shape, vtype=vtype, lb=lb, ub=ub, name=ascii_name)
                    params[name] = (var, vtype, True)

            # case bounds are ordered
            if values is None:
                for k in range(self.num_cases):
                    model.addConstr(
                        params[f'ub__{action}__{k}'][0] >= params[f'lb__{action}__{k}'][0])
                    if identity and k + 1 < self.num_cases:
                        model.addConstr(params[f'lb__{action}__{k + 1}'][0] >=
                                        params[f'ub__{action}__{k}'][0] + compiled.epsilon)
        return params

    def init_params(self, compiled: GurobiRDDLCompiler,
                    model: gurobipy.Model) -> Dict[str, Any]:
        param_values = {}
        for action in compiled.rddl.action_fluents:
            shapes = self._param_shapes(compiled, action)

            # non-overlapping case bounds, weights zero, bias the default action
            def_bounds = (-100., 100.)
            if self._is_identity(compiled, action):
                lb, ub = (np.where(np.isfinite(b), b, d) 
                          for (b, d) in zip(self._state_bounds(compiled, action), def_bounds))
            else:
                lb, ub = def_bounds
            delta = (ub - lb) / self.num_cases
            for (name, shape) in shapes.items():
                if name.startswith('lb__') or name.startswith('ub__'):
                    k = int(name.rsplit('__', 1)[-1])
                    value = lb + delta * k + compiled.epsilon if name.startswith('lb__') \
                        else lb + delta * (k + 1) - compiled.epsilon
                elif name.endswith('__bias'):
                    value = compiled.init_values[action]
                else:
                    value = 0.
                param_values[name] = np.broadcast_to(value, shape)
        return param_values

    def actions(self, compiled: GurobiRDDLCompiler,
                model: gurobipy.Model,
                params: Dict[str, Any],
                step: int,
                subs: Dict[str, Any]) -> Dict[str, Any]:
        rddl = compiled.rddl
        eps = compiled.epsilon
        action_vars = {}
        for (action, prange) in rddl.action_ranges.items():
            atype = compiled.GUROBI_TYPES.get(prange, GRB.INTEGER)
            shape = np.shape(compiled.init_values[action])
            alb, aub = self._bounds(compiled, action)
            key = f'{re.sub("[^A-z0-9_]", "", action)}at{step}'

            # convert an expression to a tensor variable, or broadcast if a constant
            def as_tensor(expr, vtype, name, lb=-GRB.INFINITY, ub=GRB.INFINITY):
                if isinstance(expr, np.ndarray) or np.isscalar(expr):
                    return np.broadcast_to(expr, shape), vtype, False
                var = model.addMVar(shape, vtype=vtype, lb=lb, ub=ub, name=name)
                model.addConstr(var == expr, name=f'C{name}')
                return var, vtype, True

            # returns whether diff >= 0, or a constant if symb is False
            def at_least(diff, symb, name):
                if not symb:
                    return np.asarray(diff >= 0, dtype=np.float64), False
                res = compiled._new_bool(model, shape, name)
                model.addConstr((res == 1) >> (diff >= 0), name=f'C{name}1')
                model.addConstr((res == 0) >> (diff <= -eps), name=f'C{name}0')
                return res, True

            cases, sats = [], []
            for k in self._cases():

                # does case k hold?
                if k != 'else':
                    if self._is_identity(compiled, action):
                        state, = self._constr_states(compiled, action)
                        value, _, symb = subs[state]
                        constr = compiled._broadcast((value, GRB.CONTINUOUS, symb), shape)
                    else:
                        expr = sum(
                            self._contract(
                                params[f'constr_weight__{action}__{k}__{state}'][0],
                                subs[state][0], len(shape))
                            for state in self._constr_states(compiled, action)
                        )
                        constr = as_tensor(expr, GRB.CONTINUOUS, f'V{key}constr{k}')
                    lb, _, lb_symb = params[f'lb__{action}__{k}']
                    ub, _, ub_symb = params[f'ub__{action}__{k}']
                    lb_sat, symb1 = at_least(
                        constr[0] - lb, constr[-1] or lb_symb, f'{key}lb{k}')
                    ub_sat, symb2 = at_least(
                        ub - constr[0], constr[-1] or ub_symb, f'{key}ub{k}')
                    if symb1 or symb2:
                        sat, *_ = compiled._logical_and(
                            model, [lb_sat, ub_sat], shape, f'{key}sat{k}')
                    else:
                        sat = np.asarray(lb_sat * ub_sat)
                    sats.append(sat)

                # case value
                expr = params[f'value_weight__{action}__{k}__bias'][0]
                for state in self.dependencies_values.get(action, []):
                    expr = expr + self._contract(
                        params[f'value_weight__{action}__{k}__{state}'][0],
                        subs[state][0], len(shape))
                cases.append(as_tensor(expr, atype, f'V{key}case{k}', alb, aub))

            # at most one case holds, else the else case: index of the case taken
            symb = any(not isinstance(sat, np.ndarray) for sat in sats)
            K = self.num_cases
            if symb:
                model.addConstr(sum(sats) <= 1, name=f'C{key}disjoint')
                index = K - sum((K - k) * sat for (k, sat) in enumerate(sats))
            
            # first case that holds, as in evaluate
            else: 
                index = np.select([sat > 0.5 for sat in sats], list(range(K)), default=K)
            action_vars[action] = compiled._select(
                model, cases, (index, GRB.INTEGER, symb), key)

        return action_vars

    def evaluate(self, compiled: GurobiRDDLCompiler,
                 params: Dict[str, Any],
                 step: int,
                 subs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        rddl = compiled.rddl
        value_of = lambda name: np.asarray(
            params[name][0].X if params[name][-1] else params[name][0])

        # evaluate each action-fluent's piecewise policy
        action_values = {}
        for (action, prange) in rddl.action_ranges.items():
            shape = np.shape(compiled.init_values[action])
            sats, values = [], []
            for k in self._cases():

                # evaluate the constraint for this case
                if k != 'else':
                    if self._is_identity(compiled, action):
                        state, = self._constr_states(compiled, action)
                        constr = np.broadcast_to(subs[state], shape)
                    else:
                        constr = sum(self._contract(
                            value_of(f'constr_weight__{action}__{k}__{state}'),
                            np.asarray(subs[state]), len(shape))
                            for state in self._constr_states(compiled, action))
                    lb, ub = value_of(f'lb__{action}__{k}'), value_of(f'ub__{action}__{k}')

                    # the model separates holds (>= 0) from not (<= -eps)
                    tol = compiled.epsilon / 2
                    sats.append((constr - lb >= -tol) & (ub - constr >= -tol))

                # evaluate the case value
                value = value_of(f'value_weight__{action}__{k}__bias')
                for state in self.dependencies_values.get(action, []):
                    value = value + self._contract(
                        value_of(f'value_weight__{action}__{k}__{state}'),
                        np.asarray(subs[state]), len(shape))
                values.append(np.broadcast_to(value, shape))

            # first case that holds, clipped to the valid range and cast
            value = np.select(sats, values[:-1], default=values[-1])
            value = np.clip(value, *self._bounds(compiled, action))
            if prange == 'bool':
                value = value > 0.5
            elif prange != 'real':
                value = np.round(value).astype(np.int64)
            action_values[action] = value
        return action_values

    def to_string(self, compiled: GurobiRDDLCompiler,
                  params: Dict[str, Any]) -> str:
        value_of = lambda name: np.asarray(
            params[name][0].X if params[name][-1] else params[name][0])
        
        # return a string representation of the piecewise policy
        lines = []
        for action in compiled.rddl.action_fluents:
            identity = self._is_identity(compiled, action)
            for k in self._cases():
                if k == 'else':
                    case = 'else'
                else:
                    constr = self._constr_states(compiled, action)[0] if identity else ' + '.join(
                        f'{value_of(f"constr_weight__{action}__{k}__{s}")} . {s}'
                        for s in self._constr_states(compiled, action)
                    )
                    case = (f'if {value_of(f"lb__{action}__{k}")} <= {constr} '
                            f'<= {value_of(f"ub__{action}__{k}")}')
                terms = [f'{value_of(f"value_weight__{action}__{k}__bias")}'] + [
                    f'{value_of(f"value_weight__{action}__{k}__{s}")} . {s}'
                    for s in self.dependencies_values.get(action, [])
                ]
                lines.append(f'{action} = {" + ".join(terms)} {case}')
        return '\n'.join(lines)


# ***********************************************************************
# ALL VERSIONS OF GUROBI CONTROLLERS
#
# - offline controller/open-loop planner
# - online controller/replanner
#
# ***********************************************************************


def _is_tensor_state(rddl: RDDLLiftedModel, state: Dict[str, Any]) -> bool:
    '''Whether state comes from a vectorized environment (lifted tensors).'''
    return (all(name in rddl.variable_ranges for name in state) and 
            any(np.ndim(value) > 0 for value in state.values()))


def _lift_state(compiler: GurobiRDDLCompiler, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
    '''Fluent tensors from an environment state, lifted or grounded.'''
    rddl = compiler.rddl
    values = {}
    for (var, groundings) in rddl.variable_groundings.items():
        if var not in compiler.init_values:
            continue
        if var in state:
            value = state[var]
        elif groundings and all(name in state for name in groundings):
            value = [state[name] for name in groundings]
        else:
            continue
        value = np.reshape(value, np.shape(compiler.init_values[var]))
        prange = rddl.variable_ranges[var]
        if prange in rddl.type_to_objects and value.dtype.type is np.str_:
            value = rddl.object_string_to_index_array(prange, value)
        values[var] = value
    return values


def _env_actions(compiler: GurobiRDDLCompiler, actions: Dict[str, np.ndarray],
                 tensors: bool) -> Dict[str, Any]:
    '''Actions in the environment's format.'''
    if tensors:
        return actions
    rddl = compiler.rddl
    grounded = {}
    for (action, value) in actions.items():
        prange = rddl.action_ranges[action]
        default = np.ravel(np.asarray(compiler.init_values[action]))
        value = np.ravel(value)
        changed = value != default
        if prange in rddl.type_to_objects:
            value = rddl.index_to_object_string_array(prange, value)
        for (name, item, differs) in zip(rddl.variable_groundings[action], value, changed):
            if differs:
                grounded[name] = item.item() if isinstance(item, np.generic) else item
    return grounded


class GurobiOfflineController(BaseAgent):
    '''A container class for a Gurobi policy trained offline.'''

    use_tensor_obs = None

    def __init__(self, rddl: RDDLLiftedModel,
                 plan: GurobiPlan,
                 env: Optional[gurobipy.Env]=None,
                 **compiler_kwargs) -> None:
        '''Creates a new Gurobi control policy that is optimized offline.

        :param rddl: the RDDL model
        :param plan: the plan or policy to optimize
        :param env: an existing gurobi environment
        :param allow_synchronous_state: whether state-fluent can be synchronous
        :param rollout_horizon: length of the planning horizon (uses the RDDL
        defined horizon if None)
        :param epsilon: small positive constant used to encode strict
        comparisons in Gurobi constraints
        :param model_params: dictionary of parameter name and values to
        pass to Gurobi model after compilation
        :param sampling: 'mean' (determinized) or 'reparam' (sampled scenarios)
        :param num_scenarios: number of sampled rollouts with sampling='reparam'
        :param seed: seed of the noise samples with sampling='reparam'
        :param verbose: whether to print nothing (0) or a summary (1) on compile
        '''
        if rddl.observ_fluents and not isinstance(plan, GurobiStraightLinePlan):
            raise RDDLNotImplementedError(
                'POMDPs are only supported with an offline GurobiStraightLinePlan: '
                'policies would read the initial state in place of observations.')
        self.rddl = rddl
        self.plan = plan
        self.compiler = GurobiRDDLCompiler(rddl=rddl, plan=plan, **compiler_kwargs)

        # optimize the plan or policy here
        self.reset()
        if env is None:
            env = gurobipy.Env()
        self.env = env
        model, _, params = self.compiler.compile(env=self.env)
        model.optimize()
        self.model = model
        self.params = params

        # check for existence of valid solution
        self.solved = model.SolCount > 0
        if not self.solved:
            raise_warning('Gurobi failed to find a feasible solution '
                          'in the given time limit: using no-op action.', 'red')

    def sample_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.solved:
            return {}

        # evaluate policy at the current time step with current inputs
        subs = {**self.compiler.init_values, **_lift_state(self.compiler, state)}
        action_values = self.plan.evaluate(
            self.compiler, params=self.params, step=self.step, subs=subs)
        self.step += 1
        return _env_actions(
            self.compiler, action_values, _is_tensor_state(self.rddl, state))

    def reset(self) -> None:
        self.step = 0


class GurobiOnlineController(BaseAgent):
    '''A container class for a Gurobi policy trained online.'''

    use_tensor_obs = None

    def __init__(self, rddl: RDDLLiftedModel,
                 plan: GurobiPlan,
                 env: Optional[gurobipy.Env]=None,
                 **compiler_kwargs) -> None:
        '''Creates a new Gurobi control policy that is optimized online.

        :param rddl: the RDDL model
        :param plan: the plan or policy to optimize
        :param env: an existing gurobi environment
        :param allow_synchronous_state: whether state-fluent can be synchronous
        :param rollout_horizon: length of the planning horizon (uses the RDDL
        defined horizon if None)
        :param epsilon: small positive constant used to encode strict
        comparisons in Gurobi constraints
        :param model_params: dictionary of parameter name and values to
        pass to Gurobi model after compilation
        :param sampling: 'mean' (determinized) or 'reparam' (sampled scenarios)
        :param num_scenarios: number of sampled rollouts with sampling='reparam'
        :param seed: seed of the noise samples with sampling='reparam'
        :param verbose: whether to print nothing (0) or a summary (1) on compile
        '''
        if rddl.observ_fluents:
            raise RDDLNotImplementedError(
                'POMDPs are not supported by the online controller: replanning '
                'would start from the initial state in place of the observations.')
        self.rddl = rddl
        self.plan = plan
        self.compiler = GurobiRDDLCompiler(rddl=rddl, plan=plan, **compiler_kwargs)

        # make the Gurobi environment
        if env is None:
            env = gurobipy.Env()
        self.env = env
        self.reset()

    def sample_action(self, state: Dict[str, Any]) -> Dict[str, Any]:

        # optimize the policy parameters at the current time step
        subs = _lift_state(self.compiler, state)
        model, _, params = self.compiler.compile(subs, env=self.env, start=self.start)
        model.optimize()
        self.solved = model.SolCount > 0

        # check for existence of solution
        if not self.solved:
            raise_warning('Gurobi failed to find a feasible solution '
                          'in the given time limit: using no-op action.', 'red')
            model.dispose()
            self.start = None
            return {}

        # evaluate policy at the current time step with current inputs
        subs = {**self.compiler.init_values, **subs}
        action_values = self.plan.evaluate(
            self.compiler, params=params, step=0, subs=subs)
        self.start = self.plan.warm_start(self.compiler, params)
        model.dispose()
        return _env_actions(
            self.compiler, action_values, _is_tensor_state(self.rddl, state))

    def reset(self) -> None:
        self.start = None
