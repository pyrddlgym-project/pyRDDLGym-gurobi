# pyRDDLGym-gurobi

Author: [Mike Gimelfarb](https://mike-gimelfarb.github.io)

This repository supports compilation of RDDL description files into Gurobi's mixed-integer (non-linear) programs, and automated planning tools for optimizing these programs in MDPs.

> [!NOTE]  
> The Gurobi planners currently determinize all stochastic variables, making it less suitable for highly stochastic problems or problems with (stochastic) dead ends.
> If you find it is not making sufficient progress on a stochastic problem, or doesn't scale well computationally to your problem, check out the [PROST planner](https://github.com/pyrddlgym-project/pyRDDLGym-prost) (for discrete spaces), the [JAX planner](https://github.com/pyrddlgym-project/pyRDDLGym-jax) (for continuous problems), or the [deep reinforcement learning wrappers](https://github.com/pyrddlgym-project/pyRDDLGym-rl).


## Contents

- [Installation](#installation)
- [Running the Basic Example](#running-the-basic-example)
- [Running from the Python API](#running-from-the-python-api)
- [Customizing Gurobi](#customizing-gurobi)
  - [Configuration File](#configuration-file)
  - [Passing Parameters Directly](#passing-parameters-directly)
- [Citing pyRDDLGym-gurobi](#citing-pyrddlgym-gurobi)

## Installation

The basic requirements are ``pyRDDLGym>=2.0`` and ``gurobipy>=10.0.0``. 
To run the basic example, you will also require ``rddlrepository>=2.0``. Everything except rddlrepository can be installed via pip:

```shell
pip install pyRDDLGym-gurobi
```

## Running the Basic Example

The basic example provided in pyRDDLGym-gurobi will run the Gurobi planner on a domain and instance of your choosing.
To run this, navigate to the install directory of pyRDDLGym-gurobi, and run:

```shell
python -m pyRDDLGym_gurobi.examples.run_plan <domain> <instance>
```

where:
- ``<domain>`` is the domain identifier as specified in rddlrepository (i.e. Wildfire_MDP_ippc2014), or a path pointing to a valid ``domain.rddl`` file
- ``<instance>`` instance is the instance identifier (i.e. 1, 2, ... 10) in rddlrepository, or a path pointing to a valid ``instance.rddl`` file

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

## Configuring pyRDDLGym-gurobi

The recommended way to manage planner settings is to write a configuration file with all the necessary hyper-parameters, which follows the same general format as for the JAX planner. Below is the basic structure of a configuration file for straight-line planning:

```shell
[Gurobi]
NonConvex=2
OutputFlag=0

[Optimizer]
method='GurobiStraightLinePlan'
method_kwargs={}
rollout_horizon=5
verbose=1
```

The configuration file can then be parsed and passed to the planner as follows:

```python
import os
from pyRDDLGym_gurobi.core.planner import load_config

# load the config
abs_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(abs_path, 'default.cfg')
controller_kwargs = load_config(config_path)

# pass the parameters to the controller and proceed as usual
controller = GurobiOnlineController(rddl=env.model, **controller_kwargs)
...
```

## Citing pyRDDLGym-gurobi

The [following citation](https://ojs.aaai.org/index.php/ICAPS/article/view/31480) describes the main ideas of the framework. Please cite it if you found it useful:

```
@inproceedings{gimelfarb2024jaxplan,
    title={JaxPlan and GurobiPlan: Optimization Baselines for Replanning in Discrete and Mixed Discrete and Continuous Probabilistic Domains},
    author={Michael Gimelfarb and Ayal Taitler and Scott Sanner},
    booktitle={34th International Conference on Automated Planning and Scheduling},
    year={2024},
    url={https://openreview.net/forum?id=7IKtmUpLEH}
}
