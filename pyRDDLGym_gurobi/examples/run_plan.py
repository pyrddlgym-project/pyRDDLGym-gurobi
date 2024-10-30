'''This example runs the Gurobi planner. 

The syntax is:

    python run_plan.py <domain> <instance> <horizon>
    
where:
    <domain> is the name of a domain located in the /Examples directory
    <instance> is the instance number
    <horizon> is a positive integer representing the lookahead horizon
'''
import os
import sys

import pyRDDLGym
from pyRDDLGym_gurobi.core.planner import GurobiOnlineController, load_config


def main(domain, instance):
    
    # create the environment
    env = pyRDDLGym.make(domain, instance, enforce_action_constraints=True)
    
    # load the config
    abs_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(abs_path, 'default.cfg') 
    controller_kwargs = load_config(config_path)  
    
    # create the controller  
    controller = GurobiOnlineController(rddl=env.model, **controller_kwargs)
    controller.evaluate(env, verbose=True, render=True)
    
    env.close()

            
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        print('python run_plan.py <domain> <instance>')
        exit(1)
    domain, instance = args[:2]
    main(domain, instance)
    
