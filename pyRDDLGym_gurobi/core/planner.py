from ast import literal_eval
import configparser
import os
import re
import sys
from typing import Any, Dict, List, Tuple, Optional

Kwargs = Dict[str, Any]

import gurobipy
from gurobipy import GRB

from pyRDDLGym.core.debug.exception import raise_warning
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.policy import BaseAgent

from pyRDDLGym_gurobi.core.compiler import GurobiRDDLCompiler

UNBOUNDED = (-GRB.INFINITY, +GRB.INFINITY)

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
# - quadratic policy
#
# ***********************************************************************
class GurobiPlan:
    '''Base class for all Gurobi compiled policies or plans.'''
    
    def __init__(self, action_bounds: Optional[Dict[str, Tuple[float, float]]]=None) -> None:
        if action_bounds is None:
            action_bounds = {}
        self.action_bounds = action_bounds
    
    def _bounds(self, rddl, action):
        if rddl.action_ranges[action] == 'bool':
            return (0, 1)
        else:
            return self.action_bounds.get(action, UNBOUNDED)
    
    def summarize_hyperparameters(self) -> None:
        pass
        
    def params(self, compiled: GurobiRDDLCompiler,
               model: gurobipy.Model,
               values: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        '''Returns the parameters of this plan/policy to be optimized.
        
        :param compiled: A gurobi compiler where the current plan is initialized
        :param model: the gurobi model instance
        :param values: if None, freeze policy parameters to these values
        '''
        raise NotImplementedError
        
    def init_params(self, compiled: GurobiRDDLCompiler,
                    model: gurobipy.Model) -> Dict[str, Any]:
        '''Return initial parameter values for the current policy class.
        
        :param compiled: A gurobi compiler where the current plan is initialized
        :param model: the gurobi model instance
        '''
        raise NotImplementedError

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
                 subs: Dict[str, Any]) -> Dict[str, Any]:
        '''Evaluates the current policy with state variables in subs.
        
        :param compiled: A gurobi compiler where the current plan is initialized
        :param params: parameter variables of the plan/policy
        :param step: the decision epoch
        :param subs: the set of fluent and non-fluent variables available at the
        current step
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
    '''A straight-line open-loop plan in Gurobi.'''
    
    def params(self, compiled: GurobiRDDLCompiler,
               model: gurobipy.Model,
               values: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        rddl = compiled.rddl
        action_vars = {}
        for (action, prange) in rddl.action_ranges.items():
            bounds = self._bounds(rddl, action)
            atype = compiled.GUROBI_TYPES[prange]
            for step in range(compiled.horizon):
                name = f'{action}__{step}'
                if values is None:
                    ascii_name = re.sub(
                        '[^A-z0-9 -]', '', action).replace(" ", "")
                    var = compiled._add_var(model, atype, *bounds,
                                            name=f'{ascii_name}at{step}')
                    action_vars[name] = (var, atype, *bounds, True)
                else:
                    value = values[name]
                    action_vars[name] = (value, atype, value, value, False)
        return action_vars
        
    def init_params(self, compiled: GurobiRDDLCompiler,
                    model: gurobipy.Model) -> Dict[str, Any]:
        param_values = {}
        for action in compiled.rddl.action_fluents:
            for step in range(compiled.horizon):
                param_values[f'{action}__{step}'] = compiled.init_values[action]
        return param_values

    def actions(self, compiled: GurobiRDDLCompiler,
                model: gurobipy.Model,
                params: Dict[str, Any],
                step: int,
                subs: Dict[str, Any]) -> Dict[str, Any]:
        action_vars = {action: params[f'{action}__{step}'] 
                       for action in compiled.rddl.action_fluents}
        return action_vars
    
    def evaluate(self, compiled: GurobiRDDLCompiler,
                 params: Dict[str, Any],
                 step: int,
                 subs: Dict[str, Any]) -> Dict[str, Any]:
        rddl = compiled.rddl
        action_values = {}
        for (action, prange) in rddl.action_ranges.items():
            name = f'{action}__{step}'
            action_value = params[name][0].X
            if prange == 'int':
                action_value = int(action_value)
            elif prange == 'bool':
                action_value = bool(action_value > 0.5)
            action_values[action] = action_value        
        return action_values
    
    def to_string(self, compiled: GurobiRDDLCompiler,
                  params: Dict[str, Any]) -> str:
        rddl = compiled.rddl
        res = ''
        for step in range(compiled.horizon):
            values = []
            for action in rddl.action_fluents:
                name = f'{action}__{step}'
                values.append(f'{action}[{step}] = {params[name][0].X}')
            res += ', '.join(values) + '\n'
        return res


class GurobiPiecewisePolicy(GurobiPlan):
    '''A piecewise linear policy in Gurobi.'''
    
    def __init__(self, *args,
                 state_bounds: Optional[Dict[str, Tuple[float, float]]]=None,
                 dependencies_constr: Optional[Dict[str, List[str]]]=None,
                 dependencies_values: Optional[Dict[str, List[str]]]=None,
                 num_cases: int=1,
                 **kwargs) -> None:
        super(GurobiPiecewisePolicy, self).__init__(*args, **kwargs)   
        if state_bounds is None:
            state_bounds = {}
        if dependencies_constr is None:
            dependencies_constr = {}
        if dependencies_values is None:
            dependencies_values = {}
        self.state_bounds = state_bounds
        self.dependencies_constr = dependencies_constr
        if dependencies_values is None:
            dependencies_values = {}
        self.dependencies_values = dependencies_values
        self.num_cases = num_cases
    
    def summarize_hyperparameters(self) -> None:
        print(f'Gurobi policy hyper-params:\n'
              f'    num_cases     ={self.num_cases}\n'
              f'    state_bounds  ={self.state_bounds}\n'
              f'    constraint_dep={self.dependencies_constr}\n'
              f'    value_dep     ={self.dependencies_values}')
        
    def _get_states_for_constraints(self, rddl):
        if self.dependencies_constr:
            return self.dependencies_constr
        else:
            state_names = list(rddl.state_fluents.keys())
            return {action: state_names for action in rddl.action_fluents}
    
    def params(self, compiled: GurobiRDDLCompiler,
               model: gurobipy.Model,
               values: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        rddl = compiled.rddl  
        states_in_constr = self._get_states_for_constraints(rddl)
        states_in_values = self.dependencies_values
        
        param_vars = {}
        for (action, arange) in rddl.action_ranges.items():
            
            # each case k
            for k in list(range(self.num_cases)) + ['else']:
                
                # constraint is a linear function of state variables
                if k != 'else':
                    
                    # constraint value assumes two cases:
                    # 1. PWL: S = {s1,... sK} --> w1 * s1 + ... + wK * sK
                    # 2. PWS: S = {s}         --> s
                    states = states_in_constr[action]
                    is_linear = len(states) > 1
                    if is_linear:
                        for feature in states:
                            name = f'constr_weight__{action}__{k}__{feature}'
                            if values is None:
                                var = compiled._add_real_var(model)
                                param_vars[name] = (var, GRB.CONTINUOUS, *UNBOUNDED, True)
                            else:
                                val = values[name]
                                param_vars[name] = (val, GRB.CONTINUOUS, val, val, False)
                        
                    # lower and upper bounds for constraint value
                    vtype = GRB.CONTINUOUS
                    lb_name = f'lb__{action}__{k}'
                    ub_name = f'ub__{action}__{k}'                    
                    if values is None:
                        lb, ub = self.state_bounds.get(states[0], UNBOUNDED)
                        var_bounds = UNBOUNDED if is_linear else (lb - 1, ub + 1)
                        lb_var = compiled._add_var(model, vtype, *var_bounds)                        
                        ub_var = compiled._add_var(model, vtype, *var_bounds)
                        model.addConstr(ub_var >= lb_var)
                        param_vars[lb_name] = (lb_var, vtype, *var_bounds, True)
                        param_vars[ub_name] = (ub_var, vtype, *var_bounds, True)
                    else:
                        lb_val = values[lb_name]       
                        ub_val = values[ub_name]                          
                        param_vars[lb_name] = (lb_val, vtype, lb_val, lb_val, False)
                        param_vars[ub_name] = (ub_val, vtype, ub_val, ub_val, False)
                    
                # action values are generally linear, but two cases:
                # C: only a bias term
                # S/L: has bias and weight parameters
                states = states_in_values.get(action, [])
                is_linear = len(states) > 0
                var_bounds = UNBOUNDED if is_linear else self._bounds(rddl, action)
                vtype = compiled.GUROBI_TYPES[arange]
                for feature in ['bias'] + states:
                    name = f'value_weight__{action}__{k}__{feature}'
                    if values is None:
                        var = compiled._add_var(model, vtype, *var_bounds)
                        param_vars[name] = (var, vtype, *var_bounds, True)
                    else:
                        val = values[name]
                        param_vars[name] = (val, vtype, val, val, False)          
                
        return param_vars
    
    def init_params(self, compiled: GurobiRDDLCompiler,
                    model: gurobipy.Model) -> Dict[str, Any]:
        rddl = compiled.rddl
        states_in_constr = self._get_states_for_constraints(rddl)
        states_in_values = self.dependencies_values
        
        param_values = {}
        for action in rddl.action_fluents:
            
            # each case k
            for k in list(range(self.num_cases)) + ['else']: 
                
                # constraint
                if k != 'else':
                    
                    # constraint value initialized to zero
                    states = states_in_constr[action]
                    is_linear = len(states) > 1
                    if is_linear:
                        for feature in states:
                            name = f'constr_weight__{action}__{k}__{feature}'
                            param_values[name] = 0
                    
                    # constraint bounds - make non-overlapping initial bounds
                    if is_linear:
                        lb, ub = -100, +100
                    else:
                        lb, ub = self.state_bounds[states[0]]
                    delta = (ub - lb) / self.num_cases
                    lbk = lb + delta * k + compiled.epsilon
                    ubk = lb + delta * (k + 1) - compiled.epsilon
                    lb_name = f'lb__{action}__{k}'
                    ub_name = f'ub__{action}__{k}'
                    param_values[lb_name] = lbk
                    param_values[ub_name] = ubk
                    
                # action value initialized to default action
                states = states_in_values.get(action, [])
                for feature in ['bias'] + states:
                    name = f'value_weight__{action}__{k}__{feature}'
                    if feature == 'bias':
                        param_values[name] = compiled.init_values[action]
                    else:
                        param_values[name] = 0
                
        return param_values
    
    def actions(self, compiled: GurobiRDDLCompiler,
                model: gurobipy.Model,
                params: Dict[str, Any],
                step: int,
                subs: Dict[str, Any]) -> Dict[str, Any]:
        rddl = compiled.rddl
        states_in_constr = self._get_states_for_constraints(rddl)
        states_in_values = self.dependencies_values
        
        action_vars = {}
        for (action, arange) in rddl.action_ranges.items():
            
            # action variable
            atype = compiled.GUROBI_TYPES[arange]
            action_bounds = self._bounds(rddl, action)
            action_var = compiled._add_var(model, atype, *action_bounds)
            action_vars[action] = (action_var, atype, *action_bounds, True)
            
            # each case k
            constr_sat_vars = []
            for k in list(range(self.num_cases)) + ['else']:
                
                # constraint
                if k != 'else':
                    
                    # constraint value
                    states = states_in_constr[action]
                    is_linear = len(states) > 1
                    if is_linear:
                        constr_value = 0
                        for feature in states:
                            name = f'constr_weight__{action}__{k}__{feature}'
                            constr_value += params[name][0] * subs[feature][0]
                        constr_value_var = compiled._add_real_var(model)
                        model.addConstr(constr_value_var == constr_value)
                    else:
                        constr_value_var = subs[states[0]][0]
                        
                    # constraint bounds
                    lb_name = f'lb__{action}__{k}'
                    ub_name = f'ub__{action}__{k}'   
                    lb_var = params[lb_name][0]
                    ub_var = params[ub_name][0]
                    lb_sat_var = compiled._add_bool_var(model)
                    ub_sat_var = compiled._add_bool_var(model)
                    lb_diff = constr_value_var - lb_var
                    ub_diff = constr_value_var - ub_var
                    model.addConstr((lb_sat_var == 1) >> (lb_diff >= 0))
                    model.addConstr((lb_sat_var == 0) >> (lb_diff <= -compiled.epsilon))
                    model.addConstr((ub_sat_var == 1) >> (ub_diff <= 0))
                    model.addConstr((ub_sat_var == 0) >> (ub_diff >= +compiled.epsilon))
                    constr_sat_var = compiled._add_bool_var(model)
                    model.addGenConstrAnd(constr_sat_var, [lb_sat_var, ub_sat_var])
                    constr_sat_vars.append(constr_sat_var)
                
                # action value
                states = states_in_values.get(action, [])
                name = f'value_weight__{action}__{k}__bias'
                action_case_value = params[name][0]
                for feature in states:
                    name = f'value_weight__{action}__{k}__{feature}'
                    action_case_value += params[name][0] * subs[feature][0]
                action_case_var = compiled._add_var(model, atype, *action_bounds)
                model.addConstr(action_case_var == action_case_value)
                
                # if the current constraint is satisfied assign action value
                if k != 'else':
                    model.addConstr((constr_sat_var == 1) >> 
                                    (action_var == action_case_var))
            
            # at most one constraint satisfied - implies disjoint case conditions
            num_sat_vars = sum(constr_sat_vars)
            model.addConstr(num_sat_vars <= 1)
            
            # if no constraint is satisfied assign default action value
            any_sat_var = compiled._add_bool_var(model)
            model.addGenConstrOr(any_sat_var, constr_sat_vars)
            model.addConstr((any_sat_var == 0) >> (action_var == action_case_var))
                    
        return action_vars
    
    def evaluate(self, compiled: GurobiRDDLCompiler,
                 params: Dict[str, Any],
                 step: int,
                 subs: Dict[str, Any]) -> Dict[str, Any]:
        rddl = compiled.rddl
        states_in_constr = self._get_states_for_constraints(rddl)
        states_in_values = self.dependencies_values
        
        action_values = {}
        for (action, arange) in rddl.action_ranges.items():
            
            # case k
            for k in list(range(self.num_cases)) + ['else']:
                
                # constraint   
                if k == 'else':
                    case_i_holds = True
                else:
                    # constraint value
                    states = states_in_constr[action]
                    is_linear = len(states) > 1
                    if is_linear:
                        constr_value = 0
                        for feature in states:
                            name = f'constr_weight__{action}__{k}__{feature}'
                            constr_value += params[name][0].X * subs[feature]
                    else:
                        constr_value = subs[states[0]]
                        
                    # constraint bounds
                    lb_name = f'lb__{action}__{k}'
                    ub_name = f'ub__{action}__{k}'
                    lb_val = params[lb_name][0].X
                    ub_val = params[ub_name][0].X
                    case_i_holds = (lb_val <= constr_value <= ub_val)
                
                # action value
                if case_i_holds:
                    states = states_in_values.get(action, [])
                    name = f'value_weight__{action}__{k}__bias'
                    action_value = params[name][0].X
                    for feature in states:
                        name = f'value_weight__{action}__{k}__{feature}'
                        action_value += params[name][0].X * subs[feature]
                    
                    # clip to valid range
                    lb, ub = self._bounds(rddl, action)
                    action_value = max(lb, min(ub, action_value))
                    
                    # cast action to appropriate type   
                    if arange == 'int':
                        action_value = int(action_value)
                    elif arange == 'bool':
                        action_value = bool(action_value > 0.5)
                    action_values[action] = action_value           
                    break
            
        return action_values

    def to_string(self, compiled: GurobiRDDLCompiler,
                  params: Dict[str, Any]) -> str:
        rddl = compiled.rddl
        states_in_constr = self._get_states_for_constraints(rddl)
        states_in_values = self.dependencies_values
        
        values = []
        for action in rddl.action_fluents:
            
            # case k
            action_case_values = []
            for k in list(range(self.num_cases)) + ['else']:
                
                # constraint
                if k != 'else':
                    
                    # constraint value
                    states = states_in_constr[action]
                    if len(states) > 1:
                        constr_values = []
                        for feature in states:
                            name = f'constr_weight__{action}__{k}__{feature}'
                            constr_values.append(f'{params[name][0].X} * {feature}')
                        constr_value = ' + '.join(constr_values)
                    else:
                        constr_value = f'{states[0]}'
                    
                    # constraint bounds
                    lb_name = f'lb__{action}__{k}'
                    ub_name = f'ub__{action}__{k}'
                    lb_val = params[lb_name][0].X
                    ub_val = params[ub_name][0].X
                    case_i_holds = f'{lb_val} <= {constr_value} <= {ub_val}'
                else:
                    case_i_holds = 'else'
                    
                # action case value
                states = states_in_values.get(action, [])
                name = f'value_weight__{action}__{k}__bias'
                action_case_terms = [f'{params[name][0].X}']
                for feature in states:
                    name = f'value_weight__{action}__{k}__{feature}'
                    action_case_terms.append(f'{params[name][0].X} * {feature}')
                action_case_value = ' + '.join(action_case_terms)
                
                # update expression for action value
                if k == 'else':
                    action_case_values.append(action_case_value)
                else:
                    action_case_values.append(f'{action_case_value} if {case_i_holds}') 
            values.append(f'{action} = ' + ' else '.join(action_case_values))
        
        return '\n'.join(values)


class GurobiQuadraticPolicy(GurobiPlan):
    '''A quadratic policy in Gurobi.'''
    
    def __init__(self, *args,
                 action_clip_value: float=100.,
                 **kwargs) -> None:
        super(GurobiQuadraticPolicy, self).__init__(*args, **kwargs)
        self.action_clip_value = action_clip_value
        
    def params(self, compiled: GurobiRDDLCompiler,
               model: gurobipy.Model,
               values: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        rddl = compiled.rddl
        states = list(rddl.state_fluents.keys())
        clip_range = (-self.action_clip_value, +self.action_clip_value)
        
        param_vars = {}
        for action in rddl.action_fluents:
            
            # linear terms
            for state in ['bias'] + states:
                name = f'weight__{action}__{state}'
                if values is None:
                    var = compiled._add_real_var(model, *clip_range)
                    param_vars[name] = (var, GRB.CONTINUOUS, *clip_range, True)
                else:
                    val = values[name]
                    param_vars[name] = (val, GRB.CONTINUOUS, val, val, False)
            
            # quadratic terms
            for (i, s1) in enumerate(states):
                for s2 in states[i:]:
                    name = f'weight__{action}__{s1}__{s2}'
                    if values is None:
                        var = compiled._add_real_var(model, *clip_range)
                        param_vars[name] = (var, GRB.CONTINUOUS, *clip_range, True)
                    else:
                        val = values[name]
                        param_vars[name] = (val, GRB.CONTINUOUS, val, val, False)
                        
        return param_vars
        
    def init_params(self, compiled: GurobiRDDLCompiler,
                    model: gurobipy.Model) -> Dict[str, Any]:
        rddl = compiled.rddl
        states = list(rddl.state_fluents.keys())
        
        param_values = {}
        for action in rddl.action_fluents:
            
            # bias initialized to no-op action value
            name = f'weight__{action}__bias'
            param_values[name] = compiled.init_values[action]
            
            # linear and quadratic terms are zero
            for state in states:
                name = f'weight__{action}__{state}'
                param_values[name] = 0
            for (i, s1) in enumerate(states):
                for s2 in states[i:]:
                    name = f'weight__{action}__{s1}__{s2}'
                    param_values[name] = 0
                    
        return param_values
    
    def actions(self, compiled: GurobiRDDLCompiler,
                model: gurobipy.Model,
                params: Dict[str, Any],
                step: int,
                subs: Dict[str, Any]) -> Dict[str, Any]:
        rddl = compiled.rddl
        states = list(rddl.state_fluents.keys())
        
        action_vars = {}        
        for action in rddl.action_fluents:
            
            # linear terms
            name = f'weight__{action}__bias'
            action_value = params[name][0]
            for state in states:
                name = f'weight__{action}__{state}'
                action_value += params[name][0] * subs[state][0]
            
            # quadratic terms
            for (i, s1) in enumerate(states):
                for s2 in states[i:]:
                    name = f'weight__{action}__{s1}__{s2}'
                    var = compiled._add_real_var(model)
                    model.addConstr(var == subs[s1][0] * subs[s2][0])
                    action_value += params[name][0] * var
            
            # action variable
            bounds = self._bounds(rddl, action)
            action_var = compiled._add_real_var(model, *bounds)
            action_vars[action] = (action_var, GRB.CONTINUOUS, *bounds, True)
            model.addConstr(action_var == action_value)
            
        return action_vars
    
    def evaluate(self, compiled: GurobiRDDLCompiler,
                 params: Dict[str, Any],
                 step: int,
                 subs: Dict[str, Any]) -> Dict[str, Any]:
        rddl = compiled.rddl
        states = list(rddl.state_fluents.keys())
        
        action_values = {}
        for action in rddl.action_fluents:
            
            # linear terms
            name = f'weight__{action}__bias'
            action_value = params[name][0].X
            for state in states:
                name = f'weight__{action}__{state}'
                action_value += params[name][0].X * subs[state]
            
            # quadratic terms
            for (i, s1) in enumerate(states):
                for s2 in states[i:]:
                    name = f'weight__{action}__{s1}__{s2}'
                    action_value += params[name][0].X * subs[s1] * subs[s2]
            
            # bound to valid range
            lb, ub = self._bounds(rddl, action)
            action_values[action] = max(min(action_value, ub), lb)
            
        return action_values
        
    def to_string(self, compiled: GurobiRDDLCompiler,
                  params: Dict[str, Any]) -> str:
        rddl = compiled.rddl
        states = list(rddl.state_fluents.keys())
        
        res = ''
        for action in rddl.action_fluents:
            
            # linear terms
            terms = []
            for state in ['bias'] + states:
                name = f'weight__{action}__{state}'
                if state == 'bias':
                    terms.append(f'{params[name][0].X}')
                else:
                    terms.append(f'{params[name][0].X} * {state}')
            
            # quadratic terms
            for (i, s1) in enumerate(states):
                for s2 in states[i:]:
                    name = f'weight__{action}__{s1}__{s2}'
                    val = params[name][0].X
                    terms.append(f'{val} * {s1} * {s2}')
            
            res += f'{action} = ' + ' + '.join(terms) + '\n'
            
        return res


# ***********************************************************************
# ALL VERSIONS OF GUROBI POLICIES
# 
# - just simple determinized planner
#
# ***********************************************************************
class GurobiOfflineController(BaseAgent):
    '''A container class for a Gurobi policy trained offline.'''
    
    def __init__(self, rddl: RDDLLiftedModel,
                 plan: GurobiPlan,
                 env: Optional[gurobipy.Env]=None,
                 **compiler_kwargs) -> None:
        '''Creates a new Gurobi control policy that is optimized offline in an 
        open-loop fashion.
        
        :param rddl: the RDDL model
        :param plan: the plan or policy to optimize
        :param env: an existing gurobi environment
        :param allow_synchronous_state: whether state-fluent can be synchronous
        :param rollout_horizon: length of the planning horizon (uses the RDDL
        defined horizon if None)
        :param epsilon: small positive constant used for comparing equality of
        real numbers in Gurobi constraints
        :param float_range: range of floating values that can be passed to 
        Gurobi to initialize fluents and non-fluents (values outside this range
        are clipped)
        :param model_params: dictionary of parameter name and values to
        pass to Gurobi model after compilation
        :param piecewise_options: a string of parameters to pass to Gurobi
        "options" parameter when creating constraints that contain piecewise
        linear approximations (e.g. cos, log, exp)
        :param logger: to log information about compilation to file
        '''
        self.rddl = rddl
        self.plan = plan
        self.compiler = GurobiRDDLCompiler(rddl=rddl, plan=plan, **compiler_kwargs)
        
        # try to use the preconditions to produce narrow action bounds
        action_bounds = self.plan.action_bounds.copy()
        for name in self.compiler.rddl.action_fluents:
            if name not in action_bounds:
                action_bounds[name] = self.compiler.bounds.get(name, UNBOUNDED)
        self.plan.action_bounds = action_bounds
        
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
        
        # inputs to the optimizer include all current fluent values
        subs = self.compiler._compile_init_subs()
        subs.update(self.compiler._compile_init_subs(state))
        subs = {name: value[0] for (name, value) in subs.items()}
        
        # check for existence of solution
        if not self.solved:
            return {}
        
        # evaluate policy at the current time step
        self.model.update()
        action_values = self.plan.evaluate(
            self.compiler, params=self.params, step=self.step, subs=subs)
        final_action_values = {}
        for (name, value) in action_values.items():
            if value != self.compiler.noop_actions[name]:
                lower, upper = self.compiler.bounds.get(name, UNBOUNDED)
                if isinstance(value, float):
                    final_action_values[name] = min(upper, max(lower, value))
                elif isinstance(value, int):
                    final_action_values[name] = int(min(upper, max(lower, value)))
                else:
                    final_action_values[name] = value                                        
        self.step += 1
        return action_values
                
    def reset(self) -> None:
        self.step = 0


class GurobiOnlineController(BaseAgent): 
    '''A container class for a Gurobi controller continuously updated using 
    state feedback.'''

    def __init__(self, rddl: RDDLLiftedModel,
                 plan: GurobiPlan,
                 env: Optional[gurobipy.Env]=None,
                 **compiler_kwargs) -> None:
        '''Creates a new Gurobi control policy that is optimized online in a 
        closed-loop fashion.
        
        :param rddl: the RDDL model
        :param plan: the plan or policy to optimize
        :param env: an existing gurobi environment
        :param allow_synchronous_state: whether state-fluent can be synchronous
        :param rollout_horizon: length of the planning horizon (uses the RDDL
        defined horizon if None)
        :param epsilon: small positive constant used for comparing equality of
        real numbers in Gurobi constraints
        :param float_range: range of floating values that can be passed to 
        Gurobi to initialize fluents and non-fluents (values outside this range
        are clipped)
        :param model_params: dictionary of parameter name and values to
        pass to Gurobi model after compilation
        :param piecewise_options: a string of parameters to pass to Gurobi
        "options" parameter when creating constraints that contain piecewise
        linear approximations (e.g. cos, log, exp)
        :param logger: to log information about compilation to file
        '''
        self.rddl = rddl
        self.plan = plan
        self.compiler = GurobiRDDLCompiler(rddl=rddl, plan=plan, **compiler_kwargs)
        
        # try to use the preconditions to produce narrow action bounds
        action_bounds = self.plan.action_bounds.copy()
        for name in self.compiler.rddl.action_fluents:
            if name not in action_bounds:
                action_bounds[name] = self.compiler.bounds.get(name, UNBOUNDED)
        self.plan.action_bounds = action_bounds
        
        # make the Gurobi environment
        if env is None:
            env = gurobipy.Env()
        self.env = env
        self.reset()
    
    def sample_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        
        # inputs to the optimizer include all current fluent values
        subs = self.compiler._compile_init_subs()
        subs.update(self.compiler._compile_init_subs(state))
        subs = {name: value[0] for (name, value) in subs.items()}     
        
        # optimize the policy parameters at the current time step
        model, _, params = self.compiler.compile(subs, env=self.env)
        model.optimize()
        self.solved = model.SolCount > 0
        
        # check for existence of solution
        if not self.solved:
            raise_warning('Gurobi failed to find a feasible solution '
                          'in the given time limit: using no-op action.', 'red')
            del model
            return {}
            
        # evaluate policy at the current time step with current inputs
        action_values = self.plan.evaluate(
            self.compiler, params=params, step=0, subs=subs)
        final_action_values = {}
        for (name, value) in action_values.items():
            if value != self.compiler.noop_actions[name]:
                lower, upper = self.compiler.bounds.get(name, UNBOUNDED)
                if isinstance(value, float):
                    final_action_values[name] = min(upper, max(lower, value))
                elif isinstance(value, int):
                    final_action_values[name] = int(min(upper, max(lower, value)))
                else:
                    final_action_values[name] = value  
        del model
        return final_action_values
        
    def reset(self) -> None:
        pass
