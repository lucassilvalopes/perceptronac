
import torch
import random
import pandas as pd
import numpy as np

from ax.core.arm import Arm
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
)

from ax.core.parameter import ParameterType, RangeParameter, ChoiceParameter
from ax.core.search_space import SearchSpace
from ax.metrics.noisy_function import NoisyFunctionMetric

# Factory methods for creating multi-objective optimization modesl.
from ax.modelbridge.factory import get_MOO_PAREGO

# Analysis utilities, including a method to evaluate hypervolumes
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.service.utils.report_utils import exp_to_df
from botorch.test_functions.multi_objective import BraninCurrin





def build_experiment(search_space,optimization_config):
    experiment = Experiment(
        name="pareto_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )
    return experiment


## Initialize with Sobol samples
def initialize_experiment(experiment,seed,n_init):
    sobol = Models.SOBOL(search_space=experiment.search_space, seed=seed)
    for _ in range(n_init):
        experiment.new_trial(sobol.gen(1)).run()    
    return experiment.fetch_data()


def get_init_hv_list(search_space,optimization_config,seed,n_init):
    dummy_experiment = build_experiment(search_space,optimization_config)
    sobol = Models.SOBOL(search_space=dummy_experiment.search_space, seed=seed)
    init_hv_list = []
    for _ in range(n_init):
        dummy_experiment.new_trial(sobol.gen(1)).run()

        dummy_model = Models.BOTORCH_MODULAR(
            experiment=dummy_experiment,
            data=dummy_experiment.fetch_data(),
        )
    
        hv = observed_hypervolume(modelbridge=dummy_model)
        init_hv_list.append(hv)

    return init_hv_list




def sobol_method(search_space,optimization_config,seed,n_init,n_batch):

    sobol_experiment = build_experiment(search_space,optimization_config)
    sobol_data = initialize_experiment(sobol_experiment,seed,n_init)

    sobol_model = Models.SOBOL(
        experiment=sobol_experiment,
        data=sobol_data,
        seed=seed
    )

    metric_names = list(sobol_experiment.metrics.keys())

    sobol_hv_list = []
    for i in range(n_batch):
        
        generator_run = sobol_model.gen(1)
        trial = sobol_experiment.new_trial(generator_run=generator_run)
        trial.run()
        exp_df = exp_to_df(sobol_experiment)
        outcomes = np.array(exp_df[metric_names], dtype=np.double)
        # Fit a GP-based model in order to calculate hypervolume.
        # We will not use this model to generate new points.
        dummy_model = Models.BOTORCH_MODULAR(
            experiment=sobol_experiment,
            data=sobol_experiment.fetch_data(),
        )
        try:
            hv = observed_hypervolume(modelbridge=dummy_model)
        except:
            hv = 0
            print("Failed to compute hv")
        sobol_hv_list.append(hv)
        print(f"Iteration: {i}, HV: {hv}")

    sobol_outcomes = np.array(exp_to_df(sobol_experiment)[metric_names], dtype=np.double)

    return sobol_hv_list



def ehvi_method(search_space,optimization_config,seed,n_init,n_batch):

    ehvi_experiment = build_experiment(search_space,optimization_config)
    ehvi_data = initialize_experiment(ehvi_experiment,seed,n_init)

    metric_names = list(ehvi_experiment.metrics.keys())

    ehvi_hv_list = []
    ehvi_model = None
    for i in range(n_batch):
        torch.manual_seed(seed)
        ehvi_model = Models.BOTORCH_MODULAR(
            experiment=ehvi_experiment,
            data=ehvi_data,
        )
        
        generator_run = ehvi_model.gen(1)
        trial = ehvi_experiment.new_trial(generator_run=generator_run)
        trial.run()
        ehvi_data = Data.from_multiple_data([ehvi_data, trial.fetch_data()])

        exp_df = exp_to_df(ehvi_experiment)
        outcomes = np.array(exp_df[metric_names], dtype=np.double)
        try:
            hv = observed_hypervolume(modelbridge=ehvi_model)
        except:
            hv = 0
            print("Failed to compute hv")
        ehvi_hv_list.append(hv)
        print(f"Iteration: {i}, HV: {hv}")

    ehvi_outcomes = np.array(exp_to_df(ehvi_experiment)[metric_names], dtype=np.double)

    return ehvi_hv_list 



def parego_method(search_space,optimization_config,seed,n_init,n_batch):

    parego_experiment = build_experiment(search_space,optimization_config)
    parego_data = initialize_experiment(parego_experiment,seed,n_init)

    metric_names = list(parego_experiment.metrics.keys())

    parego_hv_list = []
    parego_model = None
    for i in range(n_batch):

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        parego_model = get_MOO_PAREGO(
            experiment=parego_experiment,
            data=parego_data,
        )
        
        generator_run = parego_model.gen(1)
        trial = parego_experiment.new_trial(generator_run=generator_run)
        trial.run()
        parego_data = Data.from_multiple_data([parego_data, trial.fetch_data()])

        exp_df = exp_to_df(parego_experiment)
        outcomes = np.array(exp_df[metric_names], dtype=np.double)
        try:
            hv = observed_hypervolume(modelbridge=parego_model)
        except:
            hv = 0
            print("Failed to compute hv")
        parego_hv_list.append(hv)
        print(f"Iteration: {i}, HV: {hv}")

    parego_outcomes = np.array(exp_to_df(parego_experiment)[metric_names], dtype=np.double)

    return parego_hv_list