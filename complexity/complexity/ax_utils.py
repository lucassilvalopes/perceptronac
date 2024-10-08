
import os
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
from ax import OptimizationConfig

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



def build_optimization_config_mohpo(metrics,ref_point):

    mo = MultiObjective(
        objectives=[Objective(metric=metric) for metric in metrics],
    )

    objective_thresholds = [
        ObjectiveThreshold(metric=metric, bound=val, relative=False)
        for metric, val in zip(mo.metrics, ref_point)
    ]

    optimization_config = MultiObjectiveOptimizationConfig(
        objective=mo,
        objective_thresholds=objective_thresholds,
    )

    return optimization_config


def build_ax_config_objects_mohpo(parameters,metrics,data,label_to_params_func):

    search_space = SearchSpace(parameters=parameters)

    ref_point = data[[metric.name for metric in metrics]].max().values * 1.1

    optimization_config = build_optimization_config_mohpo(metrics,ref_point)

    max_hv = get_hv_from_df(search_space,optimization_config,data,label_to_params_func)

    return search_space, optimization_config, max_hv


def build_ax_config_objects_sohpo(parameters,metrics):

    search_space = SearchSpace(parameters=parameters)

    metric_a = metrics[0]

    so = Objective(metric=metric_a,minimize=True)

    optimization_config = OptimizationConfig(objective=so)

    return search_space, optimization_config


def get_true_min(data,weights):

    true_min = np.min(np.sum(np.array(weights).reshape(1,-1) * data.values,axis=1))

    return true_min


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



def get_param_lists(experiment, experiment_id):
    param_df = pd.DataFrame([trial.arm.parameters for trial in experiment.trials.values()])
    param_df.columns = [f"{experiment_id}_param_{c}" for c in param_df.columns]
    param_lists = param_df.to_dict(orient="list")
    return param_lists



def gpei_method(search_space,optimization_config,seed,n_init,n_batch):

    gpei_experiment = build_experiment(search_space,optimization_config)
    gpei_data = initialize_experiment(gpei_experiment,seed,n_init)

    for i in range(n_batch):
        torch.manual_seed(seed+i)
        gpei = Models.BOTORCH_MODULAR(
            experiment=gpei_experiment,
            data=gpei_experiment.fetch_data()
        )
        generator_run = gpei.gen(n=1)
        trial = gpei_experiment.new_trial(generator_run=generator_run)
        trial.run()
        trial.mark_completed()
    
    gpei_experiment.fetch_data()

    objective_means = np.array([[trial.objective_mean for trial in gpei_experiment.trials.values()]])

    gpei_param_lists = get_param_lists(gpei_experiment,"gpei")

    return np.minimum.accumulate(objective_means, axis=1).reshape(-1).tolist(), gpei_param_lists


def sobol_method(search_space,optimization_config,seed,n_init,n_batch):

    sobol_experiment = build_experiment(search_space,optimization_config)
    # sobol_data = initialize_experiment(sobol_experiment,seed,n_init)

    sobol_model = Models.SOBOL(
        experiment=sobol_experiment,
        # data=sobol_data,
        seed=seed
    )

    metric_names = list(sobol_experiment.metrics.keys())

    sobol_hv_list = []
    for i in range(n_init+n_batch):
        
        generator_run = sobol_model.gen(1)
        trial = sobol_experiment.new_trial(generator_run=generator_run)
        trial.run()
        sobol_experiment.fetch_data()

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

    sobol_param_lists = get_param_lists(sobol_experiment,"sobol")

    return sobol_hv_list[n_init:], sobol_param_lists



def ehvi_method(search_space,optimization_config,seed,n_init,n_batch):

    ehvi_experiment = build_experiment(search_space,optimization_config)
    ehvi_data = initialize_experiment(ehvi_experiment,seed,n_init)

    metric_names = list(ehvi_experiment.metrics.keys())

    ehvi_hv_list = []
    ehvi_model = None
    for i in range(n_batch):
        torch.manual_seed(seed+i)
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

    ehvi_param_lists = get_param_lists(ehvi_experiment,"ehvi")

    return ehvi_hv_list, ehvi_param_lists



def parego_method(search_space,optimization_config,seed,n_init,n_batch):

    parego_experiment = build_experiment(search_space,optimization_config)
    parego_data = initialize_experiment(parego_experiment,seed,n_init)

    metric_names = list(parego_experiment.metrics.keys())

    parego_hv_list = []
    parego_model = None
    for i in range(n_batch):

        torch.manual_seed(seed+i)
        random.seed(seed+i)
        np.random.seed(seed+i)
        
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

    parego_param_lists = get_param_lists(parego_experiment,"parego")

    return parego_hv_list, parego_param_lists


def df_to_trials_mohpo(df,label_to_params_func,axes):
    trials = []
    for r,c in df.iterrows():
        trials.append({
            "input":label_to_params_func(r),
            "output": { k:{"mean":c[k], "sem": 0} for k in axes }
        })
    return trials


def df_to_trials_sohpo(df,label_to_params_func,axes,weights,metric_name):
    
    trials = []
    for r,c in df.iterrows():
        trials.append({
            "input":label_to_params_func(r),
            "output": { metric_name:{"mean":sum(weights * c[axes].values), "sem": 0} }
        })
    return trials


def initialize_experiment_with_trials(exp,trials):
    for i, trial in enumerate(trials):
        arm_name = f"{i}_0"
        trial_obj = exp.new_trial()
        trial_obj.add_arm(Arm(parameters=trial["input"], name=arm_name))
        start_data = Data(df=pd.DataFrame.from_records([
                {
                    "arm_name": arm_name,
                    "metric_name": metric_name,
                    "mean": output["mean"],
                    "sem": output["sem"],
                    "trial_index": i,
                }
                for metric_name, output in trial["output"].items()
            ])
        )
        exp.attach_data(start_data)
        trial_obj.run().complete()


def get_trials_hv(search_space,optimization_config,trials):

    hv_exp = build_experiment(search_space,optimization_config)

    initialize_experiment_with_trials(hv_exp,trials)

    hv = observed_hypervolume(
        modelbridge=Models.BOTORCH_MODULAR(
            experiment=hv_exp,
            data=hv_exp.fetch_data()
        )
    )

    return hv


def get_trials_min_list(search_space,optimization_config,trials):

    min_exp = build_experiment(search_space,optimization_config)

    initialize_experiment_with_trials(min_exp,trials)
    
    min_exp.fetch_data()
    
    objective_means = np.array([[trial.objective_mean for trial in min_exp.trials.values()]])
    min_list = np.minimum.accumulate(objective_means, axis=1).reshape(-1).tolist()

    return min_list


def get_hv_from_df(search_space,optimization_config,data,label_to_params_func):

    axes = list(optimization_config.metrics.keys())

    data = data[~data.index.duplicated(keep='first')]

    trials = df_to_trials_mohpo(data,label_to_params_func,axes)

    hv = get_trials_hv(search_space,optimization_config,trials)

    return hv


def get_min_list_from_df(search_space,optimization_config,data,label_to_params_func,weights):

    axes = list(data.columns)

    metric_name = list(optimization_config.metrics.keys())[0]

    trials = df_to_trials_sohpo(data,label_to_params_func,axes,weights,metric_name)

    min_list = get_trials_min_list(search_space,optimization_config,trials)

    return min_list


def plot_min_graph(methods_df,fig_path=None):

    if "iters" in methods_df.columns:
        methods_df = methods_df.set_index("iters")

    ax =methods_df.plot(xlabel="number of observations", ylabel="Objective")
    fig = ax.get_figure()
    if fig_path is None:
        return fig
    else:
        fig.savefig(fig_path)


def plot_hv_graph(methods_df,fig_path=None):
    max_hv = methods_df["max_hv"].iloc[0]
    methods_df = methods_df.drop("max_hv",axis=1)

    if "iters" in methods_df.columns:
        methods_df = methods_df.set_index("iters")
    
    ax = (max_hv - methods_df).map(np.log10).plot(
        xlabel="number of observations", ylabel="Log Hypervolume Difference")
    fig = ax.get_figure()
    if fig_path is None:
        return fig
    else:
        fig.savefig(fig_path)



def read_csv_only_wanted_columns(fpath):
    cols = list(pd.read_csv(fpath, nrows=1))
    read_csv = pd.read_csv(fpath, usecols =[c for c in cols if ("_param_" not in c)])
    return read_csv



def avg_ax_dfs(ax_results_folder,prefix,n_iters):

    dfs = []
    for f in os.listdir(ax_results_folder):
        if f.endswith(".csv") and (prefix in f):
            dfs.append( read_csv_only_wanted_columns(os.path.join(ax_results_folder,f)) )

    common_len = min([df.shape[0] for df in dfs])

    avg_df = dfs[0].iloc[:common_len,:]
    for df in dfs[1:]:
        avg_df += df.iloc[:common_len,:]

    avg_df /= len(dfs)

    avg_df = avg_df.drop("glch_hv_list",axis=1,errors="ignore")

    if avg_df.shape[0] < n_iters:
        avg_df = avg_df.drop("iters",axis=1,errors="ignore")
        avg_df = pd.DataFrame({ 
            "iters":list(range(1,n_iters+1)),
            **{k:v+((n_iters-len(v))*[None]) for k,v in avg_df.to_dict(orient="list").items()}
        })
    elif avg_df.shape[0] > n_iters:
        avg_df = avg_df.iloc[:n_iters,:]

    else:
        pass

    return avg_df


def read_sorted_glch_data(glch_csv_path,labels_col,label_to_params_func,sort_values_by):

    glch_data = pd.read_csv(glch_csv_path)    
    glch_data = pd.concat([glch_data,glch_data[labels_col].apply(lambda x: pd.Series(label_to_params_func(x)))],axis=1)
    
    glch_data = glch_data.sort_values(by=sort_values_by).set_index(labels_col)

    return glch_data


def get_glch_hv_list(search_space,optimization_config,glch_data,label_to_params_func):

    glch_hv_list = []

    for i in range(glch_data.shape[0]):

        curr_data = glch_data.iloc[:i+1,:]

        curr_hv = get_hv_from_df(search_space,optimization_config,curr_data,label_to_params_func)
        
        glch_hv_list.append(curr_hv)
    
    return glch_hv_list


def setup_ax_loop(results_folder,n_seeds,seeds_range,n_init,n_batch):

    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    original_random_state = random.getstate()
    random.seed(42)
    random_seeds = random.sample(range(*seeds_range), n_seeds)
    random.setstate(original_random_state)

    iters = np.arange(1, n_init + n_batch + 1)

    return random_seeds, iters


def ax_loop_sohpo(results_folder,prefix,search_space,optimization_config,true_min,n_seeds,seeds_range,n_init,n_batch):

    random_seeds, iters = setup_ax_loop(results_folder,n_seeds,seeds_range,n_init,n_batch)

    for seed in random_seeds:

        gpei_min_list, gpei_param_lists = gpei_method(search_space,optimization_config,seed,n_init,n_batch)

        methods_df = pd.DataFrame({
            "iters":iters,
            "gpei_min_list": gpei_min_list,
            "true_min":len(iters)*[true_min],
            **gpei_param_lists
        }).set_index("iters")
        methods_df.to_csv(f"{results_folder}/{prefix}{seed}.csv")


def ax_loop_mohpo(results_folder,prefix,search_space,optimization_config,max_hv,n_seeds,seeds_range,n_init,n_batch):

    random_seeds, iters = setup_ax_loop(results_folder,n_seeds,seeds_range,n_init,n_batch)

    for seed in random_seeds:

        sobol_hv_list, sobol_param_lists = sobol_method(search_space,optimization_config,seed,n_init,n_batch)

        ehvi_hv_list, ehvi_param_lists = ehvi_method(search_space,optimization_config,seed,n_init,n_batch)

        parego_hv_list, parego_param_lists = parego_method(search_space,optimization_config,seed,n_init,n_batch)

        init_hv_list = get_init_hv_list(search_space,optimization_config,seed,n_init)


        methods_df = pd.DataFrame({
            "iters":iters,
            "sobol_hv_list":np.hstack([init_hv_list,sobol_hv_list]),
            "ehvi_hv_list":np.hstack([init_hv_list,ehvi_hv_list]),
            "parego_hv_list":np.hstack([init_hv_list,parego_hv_list]),
            **sobol_param_lists,
            **ehvi_param_lists,
            **parego_param_lists
        }).set_index("iters")
        methods_df["max_hv"] = max_hv


        methods_df.to_csv(f"{results_folder}/{prefix}{seed}.csv")



def custom_e_notation(number):
    p = int(np.floor(np.log10(np.abs(number))))
    c = number/10**p
    s1 = "{:2.0f}".format(10*c)
    s2 = "{:d}".format(p-1)
    return f"{s1}e{s2}"



def ax_glch_comparison_sohpo(
    results_folder,data_csv_path,setup_func,
    glch_csv_path,read_glch_data_func,label_to_params_func,
    n_seeds,seeds_range,n_init
    ):

    search_space,optimization_config,true_min,weights,prefix = setup_func(data_csv_path)

    glch_data = read_glch_data_func(glch_csv_path)

    glch_min_list = get_min_list_from_df(search_space,optimization_config,glch_data,label_to_params_func,weights)

    n_iters = len(glch_min_list)

    n_batch = n_iters - n_init

    prefix = f"{prefix}_seed"

    ax_loop_sohpo(results_folder,prefix,search_space,optimization_config,true_min,n_seeds,seeds_range,n_init,n_batch)

    avg_df = avg_ax_dfs(results_folder,prefix,n_iters)

    glch_df = pd.DataFrame({"glch_min_list":glch_min_list})

    comb_df = pd.concat([avg_df,glch_df],axis=1).set_index("iters")

    comb_df.to_csv(f"{results_folder}/{prefix.replace('_seed','_avgs')}.csv")

    plot_min_graph(comb_df,f"{results_folder}/{prefix.replace('_seed','_avgs')}.png")


def ax_glch_comparison_mohpo(
    results_folder,data_csv_path,setup_func,
    glch_csv_paths,read_glch_data_func,label_to_params_func,
    n_seeds,seeds_range,n_init
    ):

    parameters,metrics,data = setup_func(data_csv_path)

    search_space,optimization_config,max_hv = build_ax_config_objects_mohpo(parameters,metrics,data,label_to_params_func)

    glch_hv_lists = dict()
    for lbl,glch_csv_path in glch_csv_paths.items():
        glch_data = read_glch_data_func(glch_csv_path)
        glch_hv_lists[lbl] = get_glch_hv_list(search_space,optimization_config,glch_data,label_to_params_func)

    n_iters = max([len(glch_hv_list) for glch_hv_list in glch_hv_lists.values()])

    glch_hv_lists = {k:v+((n_iters-len(v))*[None]) for k,v in glch_hv_lists.items()}

    n_batch = n_iters - n_init

    prefix = f"{'_'.join(optimization_config.metrics.keys()).replace('/','_over_')}_ax_methods_seed"

    ax_loop_mohpo(results_folder,prefix,search_space,optimization_config,max_hv,n_seeds,seeds_range,n_init,n_batch)

    avg_df = avg_ax_dfs(results_folder,prefix,n_iters)

    glch_df = pd.DataFrame(glch_hv_lists)

    comb_df = pd.concat([avg_df,glch_df],axis=1).set_index("iters")

    comb_df.to_csv(f"{results_folder}/{prefix.replace('_seed','_avgs')}.csv")

    plot_hv_graph(comb_df,f"{results_folder}/{prefix.replace('_seed','_avgs')}.png")