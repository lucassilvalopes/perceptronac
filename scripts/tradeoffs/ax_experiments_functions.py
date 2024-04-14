
import random
import pandas as pd
import numpy as np

from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
)

from ax.core.parameter import ParameterType, ChoiceParameter
from ax.core.search_space import SearchSpace
from ax.metrics.noisy_function import NoisyFunctionMetric

from ax_utils import sobol_method,ehvi_method,parego_method,get_summary_df
from ax_utils import get_init_hv_list,plot_mohpo_methods, combine_results
from ax_experiments_functions import ax_rdc_setup, get_glch_hv_list_rdc

from ax_utils import get_trials_hv


def label_to_params(label):
    splitm = label.split("M")
    M = int(splitm[1])
    splitn = splitm[0].split("N")
    N = int(splitn[1])
    splitl = splitn[0].split("L")
    L = float(splitl[1])
    splitd = splitl[0].split("D")
    D = int(splitd[1])
    return {"D":D,"L":L,"N":N,"M":M}


def df_to_trials(df,complexity_axis):
    trials = []
    for r,c in df.iterrows():
        trials.append({
            "input":label_to_params(r),
            "output": {
                "bpp_loss":{"mean":c["bpp_loss"], "sem": 0},
                "mse_loss":{"mean":c["mse_loss"], "sem": 0},
                complexity_axis:{"mean":c[complexity_axis], "sem": 0}
            }
        })
    return trials


def get_glch_hv_list_rdc(search_space,optimization_config,glch_csv_path,complexity_axis):

    glch_csv = pd.read_csv(glch_csv_path)
    glch_csv[["D","L","N","M"]] = glch_csv["labels"].apply(lambda x: pd.Series(label_to_params(x)))
    glch_csv = glch_csv.sort_values(by=['iteration', 'D','N','M','L']).set_index("labels")

    glch_hv_list = []

    for i in range(glch_csv.shape[0]):

        curr_data = glch_csv.iloc[:i+1,:]

        curr_trials = df_to_trials(curr_data,complexity_axis)
        
        curr_hv = get_trials_hv(search_space,optimization_config,curr_trials)
        
        glch_hv_list.append(curr_hv)
    
    return glch_hv_list


def ax_rdc_setup(data_csv_path,complexity_axis="params"):

    data = pd.read_csv(data_csv_path).set_index("labels")

    ref_point = data[["bpp_loss","mse_loss",complexity_axis]].max().values * 1.1

    x1 = ChoiceParameter(name="D", values=[3,4], parameter_type=ParameterType.INT, is_ordered=True, sort_values=True)
    x2 = ChoiceParameter(name="L", values=[5e-3, 1e-2, 2e-2], parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True)
    x3 = ChoiceParameter(name="N", values=[32, 64, 96, 128, 160, 192, 224], parameter_type=ParameterType.INT, is_ordered=True, sort_values=True)
    x4 = ChoiceParameter(name="M", values=[32, 64, 96, 128, 160, 192, 224, 256, 288, 320], parameter_type=ParameterType.INT, is_ordered=True, 
                        sort_values=True)

    search_space = SearchSpace(parameters=[x1, x2, x3, x4])

    def params_to_label(D,L,N,M):
        D = str(int(D))
        L = "5e-3" if L == 5e-3 else ("1e-2" if L == 1e-2 else "2e-2")
        N = str(int(N))
        M = str(int(M))
        return f"D{D}L{L}N{N}M{M}"

    class MetricA(NoisyFunctionMetric):
        def f(self, x: np.ndarray) -> float:
            return float(data.loc[params_to_label(*x),"bpp_loss"])

    class MetricB(NoisyFunctionMetric):
        def f(self, x: np.ndarray) -> float:
            return float(data.loc[params_to_label(*x),"mse_loss"])

    class MetricC(NoisyFunctionMetric):
        def f(self, x: np.ndarray) -> float:
            return float(data.loc[params_to_label(*x),complexity_axis]) 


    metric_a = MetricA("bpp_loss", ["D", "L", "N", "M"], noise_sd=0.0, lower_is_better=True)
    metric_b = MetricB("mse_loss", ["D", "L", "N", "M"], noise_sd=0.0, lower_is_better=True)
    metric_c = MetricC(complexity_axis, ["D", "L", "N", "M"], noise_sd=0.0, lower_is_better=True)

    mo = MultiObjective(
        objectives=[Objective(metric=metric_a), Objective(metric=metric_b), Objective(metric=metric_c)],
    )

    objective_thresholds = [
        ObjectiveThreshold(metric=metric, bound=val, relative=False)
        for metric, val in zip(mo.metrics, ref_point)
    ]

    optimization_config = MultiObjectiveOptimizationConfig(
        objective=mo,
        objective_thresholds=objective_thresholds,
    )

    max_hv_trials = df_to_trials(data,complexity_axis)

    max_hv = get_trials_hv(search_space,optimization_config,max_hv_trials)

    return search_space, optimization_config, max_hv



def ax_rdc(data_csv_path,complexity_axis,glch_csv_path,results_folder,n_seeds,seeds_range = [1, 10000],n_init=6):

    original_random_state = random.getstate()
    random.seed(42)
    random_seeds = random.sample(range(*seeds_range), n_seeds)
    random.setstate(original_random_state)

    search_space,optimization_config,max_hv = ax_rdc_setup(data_csv_path)

    glch_hv_list = get_glch_hv_list_rdc(search_space,optimization_config,glch_csv_path,complexity_axis)

    n_batch = len(glch_hv_list) - n_init

    for seed in random_seeds:

        sobol_hv_list = sobol_method(search_space,optimization_config,seed,n_init,n_batch)

        ehvi_hv_list = ehvi_method(search_space,optimization_config,seed,n_init,n_batch)

        parego_hv_list = parego_method(search_space,optimization_config,seed,n_init,n_batch)

        init_hv_list = get_init_hv_list(search_space,optimization_config,seed,n_init)

        iters = np.arange(1, n_init + n_batch + 1)
        methods_df = get_summary_df(iters,init_hv_list,sobol_hv_list,ehvi_hv_list,parego_hv_list,len(iters)*[None],max_hv)
        methods_df.to_csv(f"{results_folder}/bpp_loss_mse_loss_{complexity_axis}_ax_methods_seed{seed}.csv")


    avg_df = combine_results(results_folder,glch_hv_list)

    plot_mohpo_methods(avg_df,f"{results_folder}/bpp_loss_mse_loss_{complexity_axis}_ax_methods_avgs.png")


