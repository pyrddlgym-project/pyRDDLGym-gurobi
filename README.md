# pyRDDLGym-gurobi

Author: [Mike Gimelfarb](https://mike-gimelfarb.github.io)

This repository supports compilation of RDDL description files into Gurobi's mixed-integer (non-linear) programs, and automated planning tools for optimizing these programs in MDPs.

> [!NOTE]  
> The Gurobi planners currently determinize all stochastic variable, making it less suitable for highly stochastic problems or problems with (stochastic) dead ends.
> If you find it is not making sufficient progress on a stochastic problem, or doesn't scale well computationally to your problem, check out the [PROST planner](https://github.com/pyrddlgym-project/pyRDDLGym-prost) (for discrete spaces), the [JAX planner](https://github.com/pyrddlgym-project/pyRDDLGym-jax) (for continuous problems), or the [deep reinforcement learning wrappers](https://github.com/pyrddlgym-project/pyRDDLGym-rl).


## Contents

- [Installation](#installation)
- [Running the Basic Example](#running-the-basic-example)
- [Running from the Python API](#running-from-the-python-api)
- [Customizing Gurobi](#customizing-gurobi)
  - [Configuration File](#configuration-file)
  - [Passing Parameters Directly](#passing-parameters-directly)

## Installation

The basic requirements are ``pyRDDLGym>=2.0`` and ``gurobipy>=10.0.1``. To run the basic example, you will also require ``rddlrepository>=2.0``. Everything can be installed (assuming Anaconda):

```shell
# Create a new conda environment
conda create -n gurobiplan python=3.11
conda activate gurobiplan
conda install pip git

# Manually install pyRDDLGym and rddlrepository
pip install git+https://github.com/pyrddlgym-project/pyRDDLGym
pip install git+https://github.com/pyrddlgym-project/rddlrepository

# Install pyRDDLGym-gurobi
pip install git+https://github.com/pyrddlgym-project/pyRDDLGym-gurobi
```

## Running the Basic Example

The basic example provided in pyRDDLGym-gurobi will run the Gurobi planner on a domain and instance of your choosing.
To run this, navigate to the install directory of pyRDDLGym-gurobi, and run:

```shell
python -m pyRDDLGym_gurobi.examples.run_plan <domain> <instance> <horizon>
```

where:
- ``<domain>`` is the domain identifier as specified in rddlrepository (i.e. Wildfire_MDP_ippc2014), or a path pointing to a valid ``domain.rddl`` file
- ``<instance>`` instance is the instance identifier (i.e. 1, 2, ... 10) in rddlrepository, or a path pointing to a valid ``instance.rddl`` file
- ``<horizon>`` is the planning lookahead horizon.

## Running from the Python API

If you are working with the Python API, you can instantiate the environment and planner however you wish:

```python
import pyRDDLGym
from pyRDDLGym_gurobi.core.planner import GurobiStraightLinePlan, GurobiOnlineController

# Create the environment
env = pyRDDLGym.make("domain name", "instance name")

# Create the planner
plan = GurobiStraightLinePlan()
controller = GurobiOnlineController(rddl=env.model, plan=plan, rollout_horizon=5)

# Run the planner
controller.evaluate(env, episodes=1, verbose=True, render=True)
```

Note, that the ``GurobiOnlineController`` is an instance of pyRDDLGym's ``BaseAgent``, so the ``evaluate()`` function can be used to streamline interaction with the environment.

## Customizing Gurobi

The Gurobi compiler and planner run using the Gurobi engine and can be configured by configuring Gurobipy. 

### Configuration File

Create a ``gurobi.env`` file in the location of your running script, and in it specify the [parameters](https://www.gurobi.com/documentation/current/refman/parameters.html) that you would like to pass to Gurobi.
For example, to instruct Gurobi to limit each optimization to 60 seconds, and to print progress during optimization to console:

```ini
TimeLimit 60
OutputFlag 1
```

### Passing Parameters Directly

Parameters can be passed as a dictionary to the ``model_params`` argument of the Gurobi controller:

```python
controller = GurobiOnlineController(rddl=env.model, plan=plan, rollout_horizon=5,
                                    model_params={'NonConvex': 2, 'OutputFlag': 1})
```

and then the controller can be used as described in the previous section.


