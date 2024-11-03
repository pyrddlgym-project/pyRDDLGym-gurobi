from copy import deepcopy
import math
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import gurobipy
from gurobipy import GRB

from pyRDDLGym.core.compiler.initializer import RDDLValueInitializer
from pyRDDLGym.core.compiler.levels import RDDLLevelAnalysis
from pyRDDLGym.core.compiler.model import RDDLLiftedModel
from pyRDDLGym.core.compiler.tracer import RDDLObjectsTracer
from pyRDDLGym.core.constraints import RDDLConstraints
from pyRDDLGym.core.debug.exception import (
    raise_warning,
    print_stack_trace,
    RDDLTypeError,
    RDDLNotImplementedError,
    RDDLUndefinedVariableError
)
from pyRDDLGym.core.debug.logger import Logger
from pyRDDLGym.core.grounder import RDDLGrounder
from pyRDDLGym.core.simulator import lngamma, RDDLSimulatorPrecompiled

if TYPE_CHECKING:
    from pyRDDLGym_gurobi.core.planner import GurobiPlan


class GurobiRDDLCompiler:
    '''Compile RDDL domains as Gurobi optimization problems.'''
    
    def __init__(self, rddl: RDDLLiftedModel,
                 plan: 'GurobiPlan',
                 allow_synchronous_state: bool=True,
                 rollout_horizon: int=None,
                 epsilon: float=1e-5,
                 float_range: Tuple[float, float]=(1e-15, 1e15),
                 model_params: Optional[Dict[str, Any]]=None,
                 piecewise_options: str='',
                 logger: Optional[Logger]=None,
                 verbose: int=1) -> None:
        '''Creates a new compiler for formulating RDDL domains + instance as 
        a Gurobi mixed-integer non-linear optimization problem. In this base
        implemenation, a fixed subset of random variables are handled by
        determinization.
        
        :param rddl: the RDDL model
        :param plan: the plan or policy to optimize
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
        :param verbose: whether to print nothing (0), summary (1),
        or detailed (2) messages to console
        '''
        if model_params is None:
            model_params = {'NonConvex': 2}
        self.plan = plan
        if rollout_horizon is None:
            rollout_horizon = rddl.horizon
        self.horizon = rollout_horizon
        self.discount = rddl.discount
        self.logger = logger
        self.verbose = int(verbose)
        
        # Gurobi-specific parameters
        self.epsilon = epsilon
        self.float_range = float_range
        self.model_params = model_params
        self.pw_options = piecewise_options
        
        # type conversion to Gurobi
        self.GUROBI_TYPES = {
            'int': GRB.INTEGER,
            'real': GRB.CONTINUOUS,
            'bool': GRB.BINARY
        }
        
        # ground out the domain
        self.rddl = RDDLGrounder(deepcopy(rddl.ast)).ground()
        
        # compile initial values
        initializer = RDDLValueInitializer(self.rddl, logger=self.logger)
        self.init_values = initializer.initialize()
        
        # compute dependency graph for CPFs and sort them by evaluation order
        sorter = RDDLLevelAnalysis(
            self.rddl, allow_synchronous_state, logger=self.logger)
        self.levels = sorter.compute_levels()     
        
        # trace expressions to cache information to be used later
        tracer = RDDLObjectsTracer(self.rddl, logger=self.logger)
        self.traced = tracer.trace()
        
        # calculate no-op actions
        self.noop_actions = {}
        for (var, values) in self.init_values.items():
            if self.rddl.variable_types[var] == 'action-fluent':
                self.noop_actions[var] = values
                
        # calculate simple bounds on actions
        simulator = RDDLSimulatorPrecompiled(
            self.rddl, 
            init_values=self.init_values, 
            levels=self.levels,
            trace_info=self.traced)
        self.bounds = RDDLConstraints(simulator).bounds
        
    def summarize_hyperparameters(self) -> None:
        print(f'Gurobi compiler hyper-params:\n'
              f'    plan              ={type(self.plan).__name__}\n'
              f'    float_range       ={self.float_range}\n'
              f'    float_equality_tol={self.epsilon}\n'
              f'    lookahead_horizon ={self.horizon}\n'
              f'    verbose           ={self.verbose}\n'
              f'Gurobi model hyper-params:\n'
              f'    user_args         ={self.model_params}\n'
              f'    user_args_pw      ={self.pw_options}')
        
    # ===========================================================================
    # main compilation subroutines
    # ===========================================================================
        
    @staticmethod
    def get_variable_info(model: gurobipy.Model) -> Dict[str, Tuple]:
        '''Returns a dictionary summarizing variables in a given Gurobi Model, 
        where the keys are variable names, and values are tuples consisting of
        the variable data type and lower and upper bounds.
        '''
        model.update()
        result = {}
        for var in model.getVars():
            result[var.VarName] = (var.VType, var.LB, var.UB)
        return result
        
    def compile(self, init_values: Optional[Dict[str, Any]]=None,
                env: Optional[gurobipy.Env]=None) -> Tuple[gurobipy.Model, List[Dict[str, Any]]]:
        '''Compiles and returns the current RDDL domain as a Gurobi optimization
        problem. Also returns action variables constructed during compilation 
        and policy parameter variables.
        
        :param init_values: override the initial values of fluents and
        non-fluents as defined in the RDDL file (if None, then the original
        values defined in the RDDL domain + instance are used instead)
        '''
        if self.verbose >= 1:
            self.summarize_hyperparameters()
            self.plan.summarize_hyperparameters()
            
        model = self._create_model(env=env)
        subs = self._compile_init_subs(init_values)
        
        params = self.plan.params(self, model) 
        objective, action_vars, _ = self._compile_rollout(
            model, self.plan, params, subs)
        model.setObjective(objective, GRB.MAXIMIZE)
        
        return model, action_vars, params
    
    def _compile_rollout(self, model, plan, params, subs, value_bounds: bool=False):
        
        # logging
        if self.logger is not None:
            message = '[info] compiling initial bound information for Gurobi:'
            for (name, (_, _, lb, ub, _)) in subs.items():
                message += f'\n\t{name}, bounds=({lb}, {ub})'
            self.logger.log(message + '\n')  
                
        objective = 0
        all_action_vars = []
        all_next_state_vars = []
        for step in range(self.horizon):
            
            # add action fluent variables to model
            action_vars = plan.actions(self, model, params, step, subs)
            all_action_vars.append(action_vars)
            subs.update(action_vars)
            
            # add action constraints
            self._compile_maxnondef_constraint(model, subs, step)
            self._compile_action_preconditions(model, subs, step)
            
            # add constraint on state for the first step
            self._compile_state_invariants(model, subs, step)
                
            # evaluate CPFs and reward
            self._compile_cpfs(model, subs, step)
            reward, (lbr, ubr) = self._compile_reward(model, subs, step)
            discount = self.discount ** step
            objective += reward * discount
            
            # update state
            all_next_state_vars.append({})
            for (state, next_state) in self.rddl.next_state.items():
                subs[state] = subs[next_state]
                all_next_state_vars[-1][next_state] = subs[next_state]
            
            # update bounds on reward
            if value_bounds:
                if step == 0:
                    lb, ub = lbr, ubr
                else:
                    lb += lbr * discount
                    ub += ubr * discount
            
            # logging
            if self.logger is not None:
                message = f'[info] compiling bound information for Gurobi at epoch {step}:'
                for (name, (_, _, lb, ub, _)) in subs.items():
                    message += f'\n\t{name}, bounds=({lb}, {ub})'
                message += f'\n\treward, bounds=({lbr}, {ubr})'
                if value_bounds:
                    message += f'\n\tvalue_fn, bounds=({lb}, {ub})'
                self.logger.log(message + '\n')  
            
        if value_bounds:
            return objective, all_action_vars, all_next_state_vars, (lb, ub)
        else:
            return objective, all_action_vars, all_next_state_vars
    
    def _create_model(self, env: gurobipy.Env=None) -> gurobipy.Model:
        if env is None:
            raise_warning(
                'Gurobi model created in default environment, not recommended.')
            model = gurobipy.Model()
        else:
            model = gurobipy.Model(env=env)
        for (name, value) in self.model_params.items():
            model.setParam(name, value)
        return model 

    def _compile_init_subs(self, init_values=None) -> Dict[str, Any]:
        if init_values is None:
            init_values = self.init_values
        rddl = self.rddl
        smallest, largest = self.float_range
        subs = {}
        for (var, value) in init_values.items():
            prange = rddl.variable_ranges[var]
            vtype = self.GUROBI_TYPES[prange]
            safe_value = value
            if rddl.variable_ranges[var] == 'real':
                if 0 < value < smallest:
                    safe_value = smallest
                elif -smallest < value < 0:
                    safe_value = -smallest
                elif value > largest:
                    safe_value = largest
                elif value < -largest:
                    safe_value = -largest
            lb, ub = GurobiRDDLCompiler._fix_bounds(safe_value, safe_value)
            subs[var] = (safe_value, vtype, lb, ub, False)
        return subs
        
    def _compile_action_preconditions(self, model, subs, step) -> None:
        for (i, precondition) in enumerate(self.rddl.preconditions):
            indicator, *_, symb = self._gurobi(precondition, model, subs, step)
            if symb:
                model.addConstr(indicator == 1, name=f'precond{i}at{step}')
    
    def _compile_state_invariants(self, model, subs, step) -> None:
        for (i, invariant) in enumerate(self.rddl.invariants):
            indicator, *_, symb = self._gurobi(invariant, model, subs, step)
            if symb:
                model.addConstr(indicator == 1, name=f'invariant{i}at{step}')
        
    def _compile_maxnondef_constraint(self, model, subs, step) -> None:
        rddl = self.rddl
        num_bool, sum_bool = 0, 0
        for (action, prange) in rddl.action_ranges.items():
            if prange == 'bool':
                var, *_ = subs[action]
                num_bool += 1
                sum_bool += var
        if rddl.max_allowed_actions < num_bool:
            model.addConstr(
                sum_bool <= rddl.max_allowed_actions, name=f'concurrent{step}')
            
    def _compile_cpfs(self, model, subs, step) -> None:
        rddl = self.rddl
        for cpfs in self.levels.values():
            for cpf in cpfs:
                _, expr = rddl.cpfs[cpf]
                subs[cpf] = self._gurobi(expr, model, subs, step)
    
    def _compile_reward(self, model, subs, step) -> tuple:
        reward, _, lb, ub, _ = self._gurobi(self.rddl.reward, model, subs, step)
        return reward, (lb, ub)
    
    # ===========================================================================
    # start of compilation subroutines
    # ===========================================================================
    
    # IMPORTANT: all helper methods below must return either a Gurobi variable
    # or a constant as the first argument
    def _gurobi(self, expr, model, subs, step):
        etype, _ = expr.etype
        if etype == 'constant':
            return self._gurobi_constant(expr, model, subs, step)
        elif etype == 'pvar':
            return self._gurobi_pvar(expr, model, subs, step)
        elif etype == 'arithmetic':
            return self._gurobi_arithmetic(expr, model, subs, step)
        elif etype == 'relational':
            return self._gurobi_relational(expr, model, subs, step)
        elif etype == 'boolean':
            return self._gurobi_logical(expr, model, subs, step)
        elif etype == 'func':
            return self._gurobi_function(expr, model, subs, step)
        elif etype == 'control':
            return self._gurobi_control(expr, model, subs, step)
        elif etype == 'randomvar':
            return self._gurobi_random(expr, model, subs, step)
        else:
            raise RDDLNotImplementedError(
                f'Expression type {etype} is not supported in Gurobi compiler.\n' + 
                print_stack_trace(expr))
            
    def _add_var(self, model, vtype, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=''):
        '''Add a generic variable to the Gurobi model.'''
        return model.addVar(vtype=vtype, lb=lb, ub=ub, name=name)
    
    def _add_bool_var(self, model, name=''):
        '''Add a BINARY variable to the Gurobi model.'''
        return self._add_var(model, GRB.BINARY, 0, 1, name=name)
    
    def _add_real_var(self, model, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=''):
        '''Add a CONTINUOUS variable to the Gurobi model.'''
        return self._add_var(model, GRB.CONTINUOUS, lb, ub, name=name)
    
    def _add_int_var(self, model, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=''):
        '''Add a INTEGER variable to the Gurobi model.'''
        return self._add_var(model, GRB.INTEGER, lb, ub, name=name)
    
    # ===========================================================================
    # leaves
    # ===========================================================================
    
    @staticmethod
    def _fix_bounds(lb, ub):
        assert (not math.isnan(lb))
        assert (not math.isnan(ub))
        assert (ub >= lb)
        lb = max(min(lb, GRB.INFINITY), -GRB.INFINITY)
        ub = max(min(ub, GRB.INFINITY), -GRB.INFINITY)
        return lb, ub
    
    @staticmethod
    def _fix_bounds_abs(lb, ub):
        if lb < 0:
            if ub <= 0:
                lb, ub = -ub, -lb
            else:
                lb, ub = 0, max(abs(lb), abs(ub))
        return GurobiRDDLCompiler._fix_bounds(lb, ub)
    
    @staticmethod
    def _fix_product_underflow(a, b):
        a1, b1 = abs(a), abs(b)
        if (a1 <= 0 or b1 <= 0) or (a1 >= 1 or b1 >= 1):
            return a * b
        tiny = np.finfo(np.float64).tiny
        if a1 < tiny / b1:
            raise_warning(f'Underflow prevented for {a} * {b} by setting to zero.')
            return 0.0
        return a * b
        
    @staticmethod
    def _fix_bounds_prod(lb1, ub1, lb2, ub2):
        lbub = (GurobiRDDLCompiler._fix_product_underflow(lb1, lb2), 
                GurobiRDDLCompiler._fix_product_underflow(lb1, ub2), 
                GurobiRDDLCompiler._fix_product_underflow(ub1, lb2), 
                GurobiRDDLCompiler._fix_product_underflow(ub1, ub2))
        return GurobiRDDLCompiler._fix_bounds(min(lbub), max(lbub))
        
    def _gurobi_constant(self, expr, model, subs, step):
        
        # get the cached value of this constant
        value = self.traced.cached_sim_info(expr)
        
        # infer type of value and assign to Gurobi type
        if isinstance(value, bool):
            vtype = GRB.BINARY
        elif isinstance(value, int):
            vtype = GRB.INTEGER
        elif isinstance(value, float):
            vtype = GRB.CONTINUOUS
        else:
            raise RDDLNotImplementedError(
                f'Range of {value} is not supported in Gurobi compiler.')
        
        # bounds form a singleton set containing the cached value
        lb, ub = GurobiRDDLCompiler._fix_bounds(value, value)
        return lb, vtype, lb, ub, False

    def _gurobi_pvar(self, expr, model, subs, step):
        var, _ = expr.args
        
        # domain object converted to canonical index
        is_value, value = self.traced.cached_sim_info(expr)
        if is_value:
            lb, ub = GurobiRDDLCompiler._fix_bounds(value, value)
            return lb, GRB.INTEGER, lb, ub, False
        
        # extract variable value
        value = subs.get(var, None)
        if value is None:
            raise RDDLUndefinedVariableError(
                f'Variable <{var}> is referenced before assignment.\n' + 
                print_stack_trace(expr))
        return value
    
    # ===========================================================================
    # arithmetic
    # ===========================================================================
    
    @staticmethod
    def _promote_vtype(vtype1, vtype2):
        if vtype1 == GRB.BINARY:
            return vtype2
        elif vtype2 == GRB.BINARY:
            return vtype1
        elif vtype1 == GRB.INTEGER:
            return vtype2
        elif vtype2 == GRB.INTEGER:
            return vtype1
        else:
            assert (vtype1 == vtype2 == GRB.CONTINUOUS)
            return vtype1
    
    @staticmethod
    def _at_least_int(vtype):
        return GurobiRDDLCompiler._promote_vtype(vtype, GRB.INTEGER)
    
    def _gurobi_arithmetic(self, expr, model, subs, step):
        _, op = expr.etype
        args = expr.args        
        n = len(args)
        
        # unary negation
        if n == 1 and op == '-':
            arg, = args
            varg, vtype, lb, ub, symb = self._gurobi(arg, model, subs, step)
            vtype = GurobiRDDLCompiler._at_least_int(vtype)
            lb, ub = GurobiRDDLCompiler._fix_bounds(-1 * ub, -1 * lb)
            
            # assign negative to a new variable
            if symb: 
                res = self._add_var(
                    model, vtype, lb, ub, name=f'E{expr.id}at{step}neg')
                model.addConstr(res + varg == 0, name=f'C{expr.id}at{step}neg')
            else:
                res = lb = ub = -1 * varg           
            return res, vtype, lb, ub, symb
        
        # binary operations
        elif n >= 1:
            results = [self._gurobi(arg, model, subs, step) for arg in args]
            
            # unwrap addition to binary operations
            if op == '+':
                sumexpr, sumvtype, sumlb, sumub, sumsymb = results[0]
                sumvtype = GurobiRDDLCompiler._at_least_int(sumvtype)
                for (varg, vtype, lb, ub, symb) in results[1:]:
                    sumexpr = sumexpr + varg
                    sumvtype = GurobiRDDLCompiler._promote_vtype(sumvtype, vtype)
                    sumlb, sumub = GurobiRDDLCompiler._fix_bounds(sumlb + lb, sumub + ub)
                    sumsymb = sumsymb or symb
                
                # assign sum to a new variable
                if sumsymb:
                    res = self._add_var(
                        model, sumvtype, sumlb, sumub, 
                        name=f'E{expr.id}at{step}add'
                    )
                    model.addConstr(
                        res == sumexpr, name=f'C{expr.id}at{step}add')
                else:
                    res = sumlb = sumub = sumexpr
                return res, sumvtype, sumlb, sumub, sumsymb
            
            # unwrap multiplication to binary operations
            elif op == '*':
                
                # accumulate the non-symbolic terms
                prodexpr = 1
                prodvtype = GRB.INTEGER
                for (varg, vtype, lb, ub, symb) in results:
                    if not symb:
                        prodexpr *= varg
                        prodvtype = GurobiRDDLCompiler._promote_vtype(prodvtype, vtype)
                
                # accumulate the symbolic terms
                prodlb = produb = prodexpr
                prodsymb = False
                for (i, (varg, vtype, lb, ub, symb)) in enumerate(results):
                    if symb:
                        prodexpr = prodexpr * varg
                        prodvtype = GurobiRDDLCompiler._promote_vtype(prodvtype, vtype)
                        prodlb, produb = GurobiRDDLCompiler._fix_bounds_prod(prodlb, produb, lb, ub)
                        prodsymb = True
                
                        # assign product to a new variable
                        prodvar = self._add_var(
                            model, prodvtype, prodlb, produb,
                            name=f'E{expr.id}at{step}mul{i}'
                        )
                        model.addConstr(
                            prodvar == prodexpr, name=f'C{expr.id}at{step}mul{i}')
                        prodexpr = prodvar
                   
                return prodexpr, prodvtype, prodlb, produb, prodsymb
            
            # subtraction
            elif n == 2 and op == '-':
                varg1, vtype1, lb1, ub1, symb1 = results[0]
                varg2, vtype2, lb2, ub2, symb2 = results[1]
                vtype = GurobiRDDLCompiler._promote_vtype(vtype1, vtype2)
                vtype = GurobiRDDLCompiler._at_least_int(vtype)
                symb = symb1 or symb2
                
                # assign difference to a new variable
                if symb:
                    lb, ub = GurobiRDDLCompiler._fix_bounds(lb1 - ub2, ub1 - lb2)
                    res = self._add_var(
                        model, vtype, lb, ub, name=f'E{expr.id}at{step}sub')
                    model.addConstr(
                        res == varg1 - varg2, name=f'C{expr.id}at{step}sub')
                else:
                    res = lb = ub = varg1 - varg2                
                return res, vtype, lb, ub, symb
            
            # implement z = x / y as a constraint z * y = x
            elif n == 2 and op == '/': 
                varg1, _, lb1, ub1, symb1 = results[0]
                varg2, _, lb2, ub2, symb2 = results[1]
                symb = symb1 or symb2
                
                if symb:
                    if symb2: 
                        if 0 > lb2 and 0 < ub2:
                            lb2, ub2 = -GRB.INFINITY, GRB.INFINITY
                        elif lb2 == 0 and ub2 == 0:
                            lb2, ub2 = GRB.INFINITY, GRB.INFINITY
                        elif lb2 == 0:
                            lb2, ub2 = 1 / ub2, GRB.INFINITY
                        elif ub2 == 0:
                            lb2, ub2 = -GRB.INFINITY, 1 / lb2
                        else:
                            lb2, ub2 = 1 / ub2, 1 / lb2
                    else:
                        lb2 = ub2 = 1 / varg2
                    lb, ub = GurobiRDDLCompiler._fix_bounds_prod(lb1, ub1, lb2, ub2)                                        
                    res = self._add_real_var(
                        model, lb, ub, name=f'E{expr.id}at{step}div')
                    model.addConstr(
                        res * varg2 == varg1, name=f'C{expr.id}at{step}div')    
                else:
                    res = lb = ub = varg1 / varg2    
                return res, GRB.CONTINUOUS, lb, ub, symb
        
        raise RDDLNotImplementedError(
            f'Arithmetic operator {op} with {n} arguments is not '
            f'supported in Gurobi compiler.\n' + 
            print_stack_trace(expr))
    
    # ===========================================================================
    # boolean
    # ===========================================================================
    
    def _gurobi_relational(self, expr, model, subs, step):
        _, op = expr.etype
        args = expr.args        
        n = len(args)
        
        if n == 2:
            lhs, rhs = args
            
            # convert <= to >=, < to >, by swapping arguments
            if op == '<=':
                lhs, rhs = rhs, lhs
                op = '>='
            elif op == '<':
                lhs, rhs = rhs, lhs
                op = '>'
            
            varg1, vtype1, lb1, ub1, symb1 = self._gurobi(lhs, model, subs, step)
            varg2, vtype2, lb2, ub2, symb2 = self._gurobi(rhs, model, subs, step)
            vtype = GurobiRDDLCompiler._promote_vtype(vtype1, vtype2)
            vtype = GurobiRDDLCompiler._at_least_int(vtype)
            symb = symb1 or symb2
                        
            # assign comparison operator to binary variable
            if op == '==' or op == '~=': 
                if symb:
                    lb, ub = GurobiRDDLCompiler._fix_bounds(lb1 - ub2, ub1 - lb2)
                    vdiff = self._add_var(
                        model, vtype, lb, ub, name=f'E{expr.id}at{step}tmp1')
                    model.addConstr(
                        vdiff == varg1 - varg2, name=f'C{expr.id}at{step}tmp1')   
                                     
                    lb, ub = GurobiRDDLCompiler._fix_bounds_abs(lb, ub)
                    vabsdiff = self._add_var(
                        model, vtype, lb, ub, name=f'E{expr.id}at{step}tmp2')
                    model.addGenConstrAbs(
                        vabsdiff, vdiff, name=f'C{expr.id}at{step}tmp2')
                    
                    if op == '==':
                        res = self._add_bool_var(
                            model, name=f'E{expr.id}at{step}eq')
                        model.addConstr(
                            (res == 1) >> (vabsdiff <= 0), 
                            name=f'C{expr.id}at{step}eq1'
                        )
                        model.addConstr(
                            (res == 0) >> (vabsdiff >= self.epsilon), 
                            name=f'C{expr.id}at{step}eq0'
                        )
                    else:
                        res = self._add_bool_var(
                            model, name=f'E{expr.id}at{step}neq')
                        model.addConstr(
                            (res == 1) >> (vabsdiff >= self.epsilon), 
                            name=f'C{expr.id}at{step}neq1'
                        )
                        model.addConstr(
                            (res == 0) >> (vabsdiff <= 0), 
                            name=f'C{expr.id}at{step}neq0'
                        )
                    lb, ub = 0, 1
                else:
                    if op == '==':
                        res = bool(varg1 == varg2)
                    else:
                        res = bool(varg1 != varg2)
                    lb = ub = int(res)
                return res, GRB.BINARY, lb, ub, symb
            
            elif op == '>=':
                if symb:
                    lb, ub = 0, 1
                    if lb1 >= ub2:
                        lb = 1
                    if ub1 <= lb2:
                        ub = 0
                    if lb == ub:
                        res = bool(lb)
                        symb = False
                    else:                        
                        res = self._add_bool_var(
                            model, name=f'E{expr.id}at{step}geq')                    
                        model.addConstr(
                            (res == 1) >> (varg1 - varg2 >= 0), 
                            name=f'C{expr.id}at{step}geq1'
                        )
                        model.addConstr(
                            (res == 0) >> (varg1 - varg2 <= -self.epsilon), 
                            name=f'C{expr.id}at{step}geq0'
                        )
                else:
                    res = bool(varg1 >= varg2)
                    lb = ub = int(res)
                return res, GRB.BINARY, lb, ub, symb
            
            elif op == '>':
                if symb:
                    lb, ub = 0, 1
                    if lb1 > ub2:
                        lb = 1
                    if ub1 < lb2:
                        ub = 0
                    if lb == ub:
                        res = bool(lb)
                        symb = False
                    else:
                        res = self._add_bool_var(
                            model, name=f'E{expr.id}at{step}gre')
                        model.addConstr(
                            (res == 1) >> (varg1 - varg2 >= self.epsilon), 
                            name=f'C{expr.id}at{step}gre1'
                        )
                        model.addConstr(
                            (res == 0) >> (varg1 - varg2 <= 0), 
                            name=f'C{expr.id}at{step}gre0'
                        )                    
                else:
                    res = bool(varg1 > varg2)
                    lb = ub = int(res)
                return res, GRB.BINARY, lb, ub, symb
            
        raise RDDLNotImplementedError(
            f'Relational operator {op} with {n} arguments is not '
            f'supported in Gurobi compiler.\n' + 
            print_stack_trace(expr))
    
    def _gurobi_logical(self, expr, model, subs, step):
        _, op = expr.etype
        if op == '&':
            op = '^'
        args = expr.args        
        n = len(args)
        
        # unary negation ~z of z is a variable y such that y + z = 1
        if n == 1 and op == '~':
            arg, = args
            varg, _, lb, ub, symb = self._gurobi(arg, model, subs, step)
            if symb:
                res = self._add_bool_var(model, name=f'E{expr.id}at{step}not')
                model.addConstr(res + varg == 1, name=f'C{expr.id}at{step}not')
                lb, ub = GurobiRDDLCompiler._fix_bounds(1 - ub, 1 - lb)
            else:
                if not isinstance(varg, (bool, np.bool_)):
                    raise RDDLTypeError(
                        f'Constant expression is of type '
                        f'{type(varg)}, expected bool or np.bool_' + 
                        '\n' + print_stack_trace(arg))
                res = not bool(varg)
                lb = ub = int(res)            
            return res, GRB.BINARY, lb, ub, symb
            
        # binary operations
        elif n >= 1:
            results = [self._gurobi(arg, model, subs, step) for arg in args]
            vargs = [result[0] for result in results]
            symbs = [result[-1] for result in results]
            symb = any(symbs)
            
            # any non-variables must be converted to variables
            if symb:
                for (i, varg) in enumerate(vargs):
                    if not symbs[i]:
                        if not isinstance(varg, (bool, np.bool_)):
                            raise RDDLTypeError(
                                f'Constant expression is of type '
                                f'{type(varg)}, expected bool or np.bool_' + 
                                '\n' + print_stack_trace(args[i]))
                        var = self._add_bool_var(
                            model, name=f'E{expr.id}at{step}tmp{i}')
                        model.addConstr(
                            var == bool(varg), name=f'C{expr.id}at{step}tmp{i}')
                        vargs[i] = var
                        symbs[i] = True
            
            # unwrap AND to binary operations
            if op == '^':
                if symb:
                    res = self._add_bool_var(model, name=f'E{expr.id}at{step}and')
                    model.addGenConstrAnd(res, vargs, name=f'C{expr.id}at{step}and')
                    lb, ub = 0, 1
                else:
                    res = all(vargs)   
                    lb = ub = int(res)                 
                return res, GRB.BINARY, lb, ub, symb
            
            # unwrap OR to binary operations
            elif op == '|':
                if symb:
                    res = self._add_bool_var(model, name=f'E{expr.id}at{step}or')
                    model.addGenConstrOr(res, vargs, name=f'C{expr.id}at{step}or')
                    lb, ub = 0, 1
                else:
                    res = any(vargs)    
                    lb = ub = int(res)                
                return res, GRB.BINARY, lb, ub, symb
            
            # unwrap => to binary operations
            elif op == '=>' and n == 2:
                varg1, varg2 = vargs
                if symb:
                    not1 = self._add_bool_var(
                        model, name=f'E{expr.id}at{step}tmp')
                    model.addConstr(
                        not1 + varg1 == 1, name=f'C{expr.id}at{step}tmp')                                                  
                    res = self._add_bool_var(
                        model, name=f'E{expr.id}at{step}imply')
                    model.addGenConstrOr(
                        res, [not1, varg2], name=f'C{expr.id}at{step}imply')
                    lb, ub = 0, 1             
                else:
                    res = (varg1 <= varg2)
                    lb = ub = int(res)
                return res, GRB.BINARY, lb, ub, symb                
                            
        raise RDDLNotImplementedError(
            f'Logical operator {op} with {n} arguments is not '
            f'supported in Gurobi compiler.\n' + 
            print_stack_trace(expr))
    
    # ===========================================================================
    # function
    # ===========================================================================

    @staticmethod
    def _log(x):
        if x <= 0:
            return -GRB.INFINITY
        else:
            return math.log(x)
    
    def _gurobi_positive(self, model, varg, vtype, lb, ub, name=''):
        lb, ub = max(lb, 0), max(ub, 0)
        res = self._add_var(model, vtype, lb, ub, name=f'E{name}')
        model.addGenConstrMax(res, [varg], constant=0, name=f'C{name}')
        return res, lb, ub
                    
    def _gurobi_function(self, expr, model, subs, step):
        _, name = expr.etype
        args = expr.args
        n = len(args)
        
        # unary functions
        if n == 1:
            arg, = args
            varg, vtype, lb, ub, symb = self._gurobi(arg, model, subs, step)
            vtype = GurobiRDDLCompiler._at_least_int(vtype)
            
            if name == 'abs': 
                if symb:
                    lb, ub = GurobiRDDLCompiler._fix_bounds_abs(lb, ub)                    
                    res = self._add_var(
                        model, vtype, lb, ub, name=f'E{expr.id}at{step}abs')
                    model.addGenConstrAbs(res, varg, name=f'C{expr.id}at{step}abs')                    
                else:
                    res = lb = ub = abs(varg)           
                return res, vtype, lb, ub, symb
            
            elif name == 'sgn':
                if symb:
                    if lb > 0:
                        res = lb = ub = 1
                        symb = False
                    elif ub < 0:
                        res = lb = ub = -1
                        symb = False
                    else:
                        pos = self._add_bool_var(
                            model, name=f'E{expr.id}at{step}tmp1')
                        model.addConstr(
                            (pos == 1) >> (varg >= self.epsilon), 
                            name=f'C{expr.id}at{step}tmp1l'
                        )
                        model.addConstr(
                            (pos == 0) >> (varg <= 0), 
                            name=f'C{expr.id}at{step}tmp1u'
                        )
                        
                        neg = self._add_bool_var(
                            model, name=f'E{expr.id}at{step}tmp2')
                        model.addConstr(
                            (neg == 1) >> (varg <= -self.epsilon), 
                            name=f'C{expr.id}at{step}tmp2u'
                        )
                        model.addConstr(
                            (neg == 0) >> (varg >= 0), 
                            name=f'C{expr.id}at{step}tmp2l'
                        )
                        
                        res = self._add_int_var(
                            model, lb=-1, ub=1, name=f'E{expr.id}at{step}sgn')
                        model.addConstr(
                            res + neg == pos, name=f'C{expr.id}at{step}sgn')
                        lb, ub = -1, 1
                else:
                    if varg > 0:
                        res = lb = ub = 1
                    elif varg < 0:
                        res = lb = ub = -1
                    else:
                        res = lb = ub = 0
                return res, GRB.INTEGER, lb, ub, symb
                
            elif name == 'floor':
                if symb:
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        math.floor(lb), math.floor(ub))
                    res = self._add_int_var(
                        model, lb, ub, name=f'E{expr.id}at{step}floor')
                    model.addConstr(
                        res <= varg, 
                        name=f'C{expr.id}at{step}flooru'
                    )
                    model.addConstr(
                        res + 1 >= varg + self.epsilon, 
                        name=f'C{expr.id}at{step}floorl'
                    )                    
                else:
                    res = lb = ub = int(math.floor(varg))
                return res, GRB.INTEGER, lb, ub, symb
            
            elif name == 'ceil':
                if symb:
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        math.ceil(lb), math.ceil(ub))
                    res = self._add_int_var(
                        model, lb, ub, name=f'E{expr.id}at{step}ceil')
                    model.addConstr(
                        res >= varg, 
                        name=f'C{expr.id}at{step}ceill'
                    )
                    model.addConstr(
                        res - 1 <= varg - self.epsilon, 
                        name=f'C{expr.id}at{step}ceilu'
                    )
                else:
                    res = lb = ub = int(math.ceil(varg))
                return res, GRB.INTEGER, lb, ub, symb
                
            elif name == 'cos':
                if symb:
                    lb, ub = -1.0, 1.0
                    res = self._add_real_var(
                        model, lb, ub, name=f'E{expr.id}at{step}cos')
                    model.addGenConstrCos(
                        varg, res, 
                        options=self.pw_options, name=f'C{expr.id}at{step}cos'
                    )
                else:
                    res = lb = ub = math.cos(varg)      
                return res, GRB.CONTINUOUS, lb, ub, symb
            
            elif name == 'sin':
                if symb:
                    lb, ub = -1.0, 1.0
                    res = self._add_real_var(
                        model, lb, ub, name=f'E{expr.id}at{step}sin')
                    model.addGenConstrSin(
                        varg, res, 
                        options=self.pw_options, name=f'C{expr.id}at{step}sin'
                    )
                else:
                    res = lb = ub = math.sin(varg)      
                return res, GRB.CONTINUOUS, lb, ub, symb
            
            elif name == 'tan':
                if symb:
                    lb, ub = -GRB.INFINITY, GRB.INFINITY
                    res = self._add_real_var(
                        model, lb, ub, name=f'E{expr.id}at{step}tan')
                    model.addGenConstrTan(
                        varg, res, 
                        options=self.pw_options, name=f'C{expr.id}at{step}tan'
                    )
                else:
                    res = lb = ub = math.tan(varg)      
                return res, GRB.CONTINUOUS, lb, ub, symb
            
            elif name == 'exp':
                if symb: 
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        math.exp(lb), math.exp(ub))
                    res = self._add_real_var(
                        model, lb, ub, name=f'E{expr.id}at{step}exp')
                    model.addGenConstrExp(
                        varg, res, 
                        options=self.pw_options, name=f'C{expr.id}at{step}exp'
                    )
                else:
                    res = lb = ub = math.exp(varg)      
                return res, GRB.CONTINUOUS, lb, ub, symb
            
            elif name == 'ln': 
                if symb:
                    arg, lb, ub = self._gurobi_positive(
                        model, varg, vtype, lb, ub, name=f'{expr.id}at{step}tmp')                    
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        GurobiRDDLCompiler._log(lb), GurobiRDDLCompiler._log(ub))
                    res = self._add_real_var(
                        model, lb, ub, name=f'E{expr.id}at{step}ln')
                    model.addGenConstrLog(
                        arg, res, 
                        options=self.pw_options, name=f'C{expr.id}at{step}ln'
                    )
                else:
                    res = lb = ub = math.log(varg)      
                return res, GRB.CONTINUOUS, lb, ub, symb
            
            elif name == 'sqrt':
                if symb: 
                    arg, lb, ub = self._gurobi_positive(
                        model, varg, vtype, lb, ub, name=f'{expr.id}at{step}tmp')                    
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        math.sqrt(lb), math.sqrt(ub))
                    res = self._add_real_var(
                        model, lb, ub, name=f'E{expr.id}at{step}sqrt')
                    model.addGenConstrPow(
                        arg, res, 0.5, 
                        options=self.pw_options, name=f'C{expr.id}at{step}sqrt'
                    )
                else:
                    res = lb = ub = math.sqrt(varg)     
                return res, GRB.CONTINUOUS, lb, ub, symb
        
        # binary functions
        elif n == 2:
            arg1, arg2 = args
            varg1, vtype1, lb1, ub1, symb1 = self._gurobi(arg1, model, subs, step)
            varg2, vtype2, lb2, ub2, symb2 = self._gurobi(arg2, model, subs, step)
            vtype = GurobiRDDLCompiler._promote_vtype(vtype1, vtype2)
            vtype = GurobiRDDLCompiler._at_least_int(vtype)
            symb = symb1 or symb2
            
            if name == 'min': 
                if symb: 
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        min(lb1, lb2), min(ub1, ub2))
                    res = self._add_var(
                        model, vtype, lb, ub, name=f'E{expr.id}at{step}min')
                    model.addGenConstrMin(
                        res, [varg1, varg2], name=f'C{expr.id}at{step}min')
                else:
                    res = lb = ub = min(varg1, varg2)  
                return res, vtype, lb, ub, symb
            
            elif name == 'max':
                if symb:
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        max(lb1, lb2), max(ub1, ub2))
                    res = self._add_var(
                        model, vtype, lb, ub, name=f'E{expr.id}at{step}max')
                    model.addGenConstrMax(
                        res, [varg1, varg2], name=f'C{expr.id}at{step}max')
                else:
                    res = lb = ub = max(varg1, varg2)  
                return res, vtype, lb, ub, symb
            
            elif name == 'pow':
                if symb: 
                    # argument must be non-negative
                    base, lb1, ub1 = self._gurobi_positive(
                        model, varg1, vtype1, lb1, ub1, 
                        name=f'{expr.id}at{step}tmp'
                    )   
                                        
                    # compute bounds on pow
                    loglb = GurobiRDDLCompiler._log(lb1)
                    logub = GurobiRDDLCompiler._log(ub1)                    
                    loglu = (loglb * lb2, loglb * ub2, logub * lb2, logub * ub2)
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        math.exp(min(loglu)), math.exp(max(loglu)))
                    
                    # assign pow to new variable
                    res = self._add_real_var(
                        model, lb, ub, name=f'E{expr.id}at{step}pow')
                    model.addGenConstrPow(
                        base, res, varg2, 
                        options=self.pw_options, name=f'C{expr.id}at{step}pow'
                    )
                else:
                    res = lb = ub = math.pow(varg1, varg2)         
                return res, GRB.CONTINUOUS, lb, ub, symb
            
            elif name == 'mod':
                if symb:
                    # second argument must be non-negative
                    varg2, lb2, ub2 = self._gurobi_positive(
                        model, varg2, vtype2, lb2, ub2, 
                        name=f'{expr.id}at{step}tmp1'
                    )  
                    
                    # compute r = x % y as x = y * q + r where 0 <= r < y
                    lb, ub = 0, max(0, ub2 - 1)
                    res = self._add_int_var(
                        model, lb, ub, name=f'E{expr.id}at{step}mod')
                    quotient = self._add_int_var(
                        model, name=f'E{expr.id}at{step}tmp2')
                    model.addConstr(
                        varg1 == varg2 * quotient + res, 
                        name=f'C{expr.id}at{step}mod'
                    )                    
                else:
                    res = lb = ub = varg1 % varg2
                return res, GRB.INTEGER, lb, ub, symb
            
            elif name == 'fmod':
                if symb:
                    # second argument must be non-negative
                    varg2, lb2, ub2 = self._gurobi_positive(
                        model, varg2, vtype2, lb2, ub2, 
                        name=f'{expr.id}at{step}tmp1'
                    )  
                    
                    # compute r = x % y as x = y * q + r where 0 <= r < y
                    lb, ub = 0, max(0, ub2 - self.epsilon)
                    res = self._add_real_var(
                        model, lb, ub, name=f'E{expr.id}at{step}fmod')
                    quotient = self._add_int_var(
                        model, name=f'E{expr.id}at{step}tmp2')
                    model.addConstr(
                        varg1 == varg2 * quotient + res, 
                        name=f'C{expr.id}at{step}fmod'
                    )                    
                else:
                    res = lb = ub = varg1 % varg2
                return res, GRB.CONTINUOUS, lb, ub, symb
            
            elif name == 'hypot':
                if symb:
                    if lb1 >= 0 or ub1 <= 0:
                        lb1, ub1 = min(lb1 ** 2, ub1 ** 2), max(lb1 ** 2, ub1 ** 2)
                    else:
                        lb1, ub1 = 0.0, max(lb1 ** 2, ub1 ** 2)
                    if lb2 >= 0 or ub2 <= 0:
                        lb2, ub2 = min(lb2 ** 2, ub2 ** 2), max(lb2 ** 2, ub2 ** 2)
                    else:
                        lb2, ub2 = 0.0, max(lb2 ** 2, ub2 ** 2)
                    lb, ub = GurobiRDDLCompiler._fix_bounds(lb1 + lb2, ub1 + ub2)
                    ssq = self._add_real_var(
                        model, lb, ub, name=f'E{expr.id}at{step}tmp')
                    model.addConstr(
                        ssq == varg1 * varg1 + varg2 * varg2, 
                        name=f'C{expr.id}at{step}tmp'
                    )        
                                
                    lb, ub = GurobiRDDLCompiler._fix_bounds(
                        math.sqrt(lb), math.sqrt(ub))
                    res = self._add_real_var(
                        model, lb, ub, name=f'E{expr.id}at{step}hypot')
                    model.addGenConstrPow(
                        ssq, res, 0.5, 
                        options=self.pw_options, name=f'C{expr.id}at{step}hypot'
                    )
                else:
                    res = lb = ub = math.sqrt(varg1 ** 2 + varg2 ** 2)
                return res, GRB.CONTINUOUS, lb, ub, symb
                
        raise RDDLNotImplementedError(
            f'Function operator {name} with {n} arguments is not '
            f'supported in Gurobi compiler.\n' + 
            print_stack_trace(expr))

    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _gurobi_control(self, expr, model, subs, step):
        _, op = expr.etype
        args = expr.args
        n = len(args)
        
        if n == 3 and op == 'if':
            pred, arg1, arg2 = args
            vargp, *_, symbp = self._gurobi(pred, model, subs, step)
            varg1, vtype1, lb1, ub1, symb1 = self._gurobi(arg1, model, subs, step)
            varg2, vtype2, lb2, ub2, symb2 = self._gurobi(arg2, model, subs, step)
            vtype = GurobiRDDLCompiler._promote_vtype(vtype1, vtype2)
            
            # assign if to new variable
            if symbp:
                lb, ub = GurobiRDDLCompiler._fix_bounds(min(lb1, lb2), max(ub1, ub2))
                res = self._add_var(
                    model, vtype, lb, ub, name=f'E{expr.id}at{step}if')
                model.addConstr(
                    (vargp == 1) >> (res == varg1), 
                    name=f'C{expr.id}at{step}if1'
                )
                model.addConstr(
                    (vargp == 0) >> (res == varg2), 
                    name=f'C{expr.id}at{step}if0'
                )
                symb = True
            else:
                if not isinstance(vargp, (bool, np.bool_)):
                    raise RDDLTypeError(
                        f'Constant expression is of type '
                        f'{type(vargp)}, expected bool or np.bool_' + 
                        '\n' + print_stack_trace(pred))
                if bool(vargp):
                    res, lb, ub, symb = varg1, lb1, ub1, symb1
                else:
                    res, lb, ub, symb = varg2, lb2, ub2, symb2
            return res, vtype, lb, ub, symb
            
        raise RDDLNotImplementedError(
            f'Control flow {op} with {n} arguments is not '
            f'supported in Gurobi compiler.\n' + 
            print_stack_trace(expr))

    # ===========================================================================
    # random variables
    # ===========================================================================
    
    def _gurobi_random(self, expr, model, subs, step):
        _, name = expr.etype
        if name == 'KronDelta':
            return self._gurobi_kron(expr, model, subs, step)
        elif name == 'DiracDelta':
            return self._gurobi_dirac(expr, model, subs, step)
        elif name == 'Uniform':
            return self._gurobi_uniform(expr, model, subs, step)
        elif name == 'Bernoulli':
            return self._gurobi_bernoulli(expr, model, subs, step)
        elif name == 'Normal':
            return self._gurobi_normal(expr, model, subs, step)
        elif name == 'Poisson':
            return self._gurobi_poisson(expr, model, subs, step)
        elif name == 'Exponential':
            return self._gurobi_exponential(expr, model, subs, step)
        elif name == 'Gamma':
            return self._gurobi_gamma(expr, model, subs, step)
        elif name == 'Weibull':
            return self._gurobi_weibull(expr, model, subs, step)
        else:
            raise RDDLNotImplementedError(
                f'Distribution {name} is not supported in Gurobi compiler.\n' + 
                print_stack_trace(expr))
    
    def _gurobi_kron(self, expr, model, subs, step): 
        arg, = expr.args
        return self._gurobi(arg, model, subs, step)
    
    def _gurobi_dirac(self, expr, model, subs, step):
        arg, = expr.args
        return self._gurobi(arg, model, subs, step)
    
    def _gurobi_uniform(self, expr, model, subs, step):
        if self.verbose >= 2:
            raise_warning('Using the replacement rule: '
                          'Uniform(a, b) --> (a + b) / 2.')
            
        arg1, arg2 = expr.args
        varg1, _, lb1, ub1, symb1 = self._gurobi(arg1, model, subs, step)
        varg2, _, lb2, ub2, symb2 = self._gurobi(arg2, model, subs, step)
        
        # determinize uniform as (lower + upper) / 2        
        symb = symb1 or symb2
        midexpr = (varg1 + varg2) / 2
        if symb:
            lb, ub = GurobiRDDLCompiler._fix_bounds(
                (lb1 + lb2) / 2, (ub1 + ub2) / 2)
            res = self._add_real_var(
                model, lb, ub, name=f'E{expr.id}at{step}unif')
            model.addConstr(res == midexpr, name=f'C{expr.id}at{step}unif')            
        else:
            res = lb = ub = midexpr
        return res, GRB.CONTINUOUS, lb, ub, symb
        
    def _gurobi_bernoulli(self, expr, model, subs, step):
        if self.verbose >= 2:
            raise_warning('Using the replacement rule: Bernoulli(p) --> p')
            
        arg, = expr.args
        varg, _, lb, ub, symb = self._gurobi(arg, model, subs, step)
        
        # determinize bernoulli as indicator of p > 0.5
        if symb:
            res = self._add_bool_var(model, name=f'E{expr.id}at{step}bern')
            model.addConstr(
                (res == 1) >> (varg >= 0.5 + self.epsilon), 
                name=f'C{expr.id}at{step}bern1'
            )
            model.addConstr(
                (res == 0) >> (varg <= 0.5), 
                name=f'C{expr.id}at{step}bern0'
            )
            lb, ub = 0, 1
        else:
            res = bool(varg > 0.5)
            lb = ub = int(res)
        return res, GRB.BINARY, lb, ub, symb
        
    def _gurobi_normal(self, expr, model, subs, step):
        if self.verbose >= 2:
            raise_warning('Using the replacement rule: Normal(m, v) --> m')
            
        mean, _ = expr.args
        varg, _, lb, ub, symb = self._gurobi(mean, model, subs, step)
        
        # determinize Normal as mean
        return varg, GRB.CONTINUOUS, lb, ub, symb
    
    def _gurobi_poisson(self, expr, model, subs, step):
        if self.verbose >= 2:
            raise_warning('Using the replacement rule: Poisson(l) --> l')
            
        rate, = expr.args
        varg, _, lb, ub, symb = self._gurobi(rate, model, subs, step)
        
        # determinize Poisson as rate
        return varg, GRB.CONTINUOUS, lb, ub, symb
    
    def _gurobi_exponential(self, expr, model, subs, step):
        if self.verbose >= 2:
            raise_warning('Using the replacement rule: Exponential(l) --> l')
            
        scale, = expr.args
        varg, _, lb, ub, symb = self._gurobi(scale, model, subs, step)
        
        # determinize Exponential as scale
        return varg, GRB.CONTINUOUS, lb, ub, symb

    def _gurobi_gamma(self, expr, model, subs, step):
        if self.verbose >= 2:
            raise_warning('Using the replacement rule: Gamma(r, s) --> r * s')
            
        shape, scale = expr.args
        varg1, _, lb1, ub1, symb1 = self._gurobi(shape, model, subs, step)
        varg2, _, lb2, ub2, symb2 = self._gurobi(scale, model, subs, step)
        
        # determinize gamma as shape * scale
        prodexpr = varg1 * varg2
        symb = symb1 or symb2
        if symb:
            lb, ub = GurobiRDDLCompiler._fix_bounds_prod(lb1, ub1, lb2, ub2)
            res = self._add_real_var(
                model, lb, ub, name=f'E{expr.id}at{step}gamma')
            model.addConstr(res == prodexpr, name=f'C{expr.id}at{step}gamma')
        else:
            res = lb = ub = prodexpr     
        return res, GRB.CONTINUOUS, lb, ub, symb
    
    def _gurobi_weibull(self, expr, model, subs, step):
        if self.verbose >= 2:
            raise_warning('Using the replacement rule: '
                          'Weibull(s, l) --> l * Gamma(1 + 1/s)')
        
        shape, scale = expr.args
        varg1, _, _, _, symb1 = self._gurobi(shape, model, subs, step)
        varg2, _, lb2, ub2, symb2 = self._gurobi(scale, model, subs, step)
        
        # check shape is non symbolic
        if symb1:
            raise RDDLNotImplementedError(
                'Weibull symbolic expression for shape parameter is not '
                'supported in Gurobi compiler.\n' + 
                print_stack_trace(expr))
        
        # estimate mean as scale * Gamma(1 + 1/shape)
        gampart = np.exp(lngamma(1. + 1. / varg1))
        gampart = max(min(gampart, GRB.INFINITY), -GRB.INFINITY)
        prodexpr = varg2 * gampart
        symb = symb1 or symb2
        if symb:
            lb, ub = GurobiRDDLCompiler._fix_bounds_prod(lb2, ub2, gampart, gampart)
            res = self._add_real_var(model, lb, ub, name=f'E{expr.id}at{step}weib')
            model.addConstr(res == prodexpr, name=f'C{expr.id}at{step}weib')
        else:
            res = lb = ub = prodexpr
        return res, GRB.CONTINUOUS, lb, ub, symb
        