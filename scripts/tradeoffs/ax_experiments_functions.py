
import random
import pandas as pd
import numpy as np
from typing import List

from ax.core.objective import MultiObjective, Objective
from ax.core.optimization_config import (
    MultiObjectiveOptimizationConfig,
    ObjectiveThreshold,
)

from ax.core.parameter import ParameterType, ChoiceParameter, FixedParameter
from ax.core.search_space import SearchSpace
from ax.metrics.noisy_function import NoisyFunctionMetric

from ax_utils import build_ax_config_objects_mohpo, read_sorted_glch_data, build_ax_config_objects_sohpo, custom_e_notation, get_true_min




def rdc_params_to_label(D,L,N,M):
    D = str(int(D))
    L = "5e-3" if L == 5e-3 else ("1e-2" if L == 1e-2 else "2e-2")
    N = str(int(N))
    M = str(int(M))
    return f"D{D}L{L}N{N}M{M}"

def rdc_label_to_params(label):
    splitm = label.split("M")
    M = int(splitm[1])
    splitn = splitm[0].split("N")
    N = int(splitn[1])
    splitl = splitn[0].split("L")
    L = float(splitl[1])
    splitd = splitl[0].split("D")
    D = int(splitd[1])
    return {"D":D,"L":L,"N":N,"M":M}

def rdc_setup(data_csv_path,complexity_axis="params"):

    data = pd.read_csv(data_csv_path).set_index("labels")

    x1 = ChoiceParameter(name="D", values=[3,4], parameter_type=ParameterType.INT, is_ordered=True, sort_values=True)
    x2 = ChoiceParameter(name="L", values=[5e-3, 1e-2, 2e-2], parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True)
    x3 = ChoiceParameter(name="N", values=[32, 64, 96, 128, 160, 192, 224], parameter_type=ParameterType.INT, is_ordered=True, sort_values=True)
    x4 = ChoiceParameter(name="M", values=[32, 64, 96, 128, 160, 192, 224, 256, 288, 320], parameter_type=ParameterType.INT, is_ordered=True, 
                        sort_values=True)

    parameters=[x1, x2, x3, x4]

    class MetricA(NoisyFunctionMetric):
        def f(self, x: np.ndarray) -> float:
            return float(data.loc[rdc_params_to_label(*x),"bpp_loss"])

    class MetricB(NoisyFunctionMetric):
        def f(self, x: np.ndarray) -> float:
            return float(data.loc[rdc_params_to_label(*x),"mse_loss"])

    class MetricC(NoisyFunctionMetric):
        def f(self, x: np.ndarray) -> float:
            return float(data.loc[rdc_params_to_label(*x),complexity_axis]) 


    metric_a = MetricA("bpp_loss", ["D", "L", "N", "M"], noise_sd=0.0, lower_is_better=True)
    metric_b = MetricB("mse_loss", ["D", "L", "N", "M"], noise_sd=0.0, lower_is_better=True)
    metric_c = MetricC(complexity_axis, ["D", "L", "N", "M"], noise_sd=0.0, lower_is_better=True)

    metrics = [metric_a,metric_b,metric_c]

    return build_ax_config_objects_mohpo(parameters,metrics,data,rdc_label_to_params)

def rdc_read_glch_data(glch_csv_path,complexity_axis="params"):
    return read_sorted_glch_data(
        glch_csv_path,"labels",rdc_label_to_params,['iteration', 'D','N','M','L'])[["bpp_loss","mse_loss",complexity_axis]]




def rdc_loss_load_data(data_csv_path,lambdas=[],complexity_axis="params"):
    data = pd.read_csv(data_csv_path)
    if len(lambdas) == 0:
        data = data.set_index("labels")
    else:
        data = data[data["labels"].apply(lambda x: any([(lmbd in x) for lmbd in lambdas]) )].set_index("labels")
    data = data[["bpp_loss","mse_loss",complexity_axis]]
    return data


def rdc_loss_prefix(data,weights,lambdas):
    formatted_axes = '_'.join(list(data.columns)).replace('/','_over_')
    formatted_lambdas = \
        "" if len(lambdas)==0 else "lambdas_" + "-".join([lambdas[i] for i in np.argsort(list(map(float,lambdas)))])+"_"
    formatted_weights = 'weights_'.join([custom_e_notation(w) for w in weights])
    prefix = f"{formatted_axes}_{formatted_weights}_{formatted_lambdas}_ax_methods"
    return prefix


def rdc_loss_setup(data_csv_path:str,weights:List[float],lambdas:List[str],complexity_axis:str):

    data = rdc_loss_load_data(data_csv_path,lambdas,complexity_axis)

    if weights is None:
        weights = [1 for _ in range(3)]

    x1 = ChoiceParameter(name="D", values=[3,4], parameter_type=ParameterType.INT, is_ordered=True, sort_values=True)
    if len(lambdas) == 1:
        x2 = FixedParameter(name="L", value=float(lambdas[0]), parameter_type=ParameterType.FLOAT)
    elif len(lambdas) > 1: 
        x2 = ChoiceParameter(name="L", values=sorted(map(float,lambdas)), parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True)
    else:
        x2 = ChoiceParameter(name="L", values=[5e-3, 1e-2, 2e-2], parameter_type=ParameterType.FLOAT, is_ordered=True, sort_values=True)
    x3 = ChoiceParameter(name="N", values=[32, 64, 96, 128, 160, 192, 224], parameter_type=ParameterType.INT, is_ordered=True, sort_values=True)
    x4 = ChoiceParameter(name="M", values=[32, 64, 96, 128, 160, 192, 224, 256, 288, 320], parameter_type=ParameterType.INT, is_ordered=True, 
                        sort_values=True)

    parameters=[x1, x2, x3, x4]

    class MetricA(NoisyFunctionMetric):
        def f(self, x: np.ndarray) -> float:
            return float(sum(np.array(weights) * data.loc[rdc_params_to_label(*x),:].values))

    metric_a = MetricA("rdc_loss", ["D", "L", "N", "M"], noise_sd=0.0, lower_is_better=True)

    metrics = [metric_a]

    true_min = get_true_min(data,weights)

    prefix = rdc_loss_prefix(data,weights,lambdas)

    return *build_ax_config_objects_sohpo(parameters,metrics),true_min,weights,prefix




def rc_params_to_label(h1,h2):
    widths = [32,h1,h2,1]
    return '_'.join(map(lambda x: f"{x:03d}",widths))

def rc_label_to_params(label):
    split_label = label.split("_")
    h1 = int(split_label[1])
    h2 = int(split_label[2])
    return {"h1":h1,"h2":h2}

def rc_setup(csv_path,complexity_axis="micro_joules_per_pixel"):

    data = pd.read_csv(csv_path).set_index("topology")

    x1 = ChoiceParameter(name="h1", values=[10,20,40,80,160,320,640], parameter_type=ParameterType.INT, is_ordered=True, sort_values=True)
    x2 = ChoiceParameter(name="h2", values=[10,20,40,80,160,320,640], parameter_type=ParameterType.INT, is_ordered=True, sort_values=True)

    parameters=[x1, x2]

    class MetricA(NoisyFunctionMetric):
        def f(self, x: np.ndarray) -> float:
            return float(data.loc[rc_params_to_label(*x),complexity_axis])

    class MetricB(NoisyFunctionMetric):
        def f(self, x: np.ndarray) -> float:
            return float(data.loc[rc_params_to_label(*x),"data_bits/data_samples"])

    metric_a = MetricA(complexity_axis, ["h1", "h2"], noise_sd=0.0, lower_is_better=True)
    metric_b = MetricB("data_bits/data_samples", ["h1", "h2"], noise_sd=0.0, lower_is_better=True)

    metrics = [metric_a,metric_b]

    return build_ax_config_objects_mohpo(parameters,metrics,data,rc_label_to_params)

def rc_read_glch_data(glch_csv_path):
    return read_sorted_glch_data(glch_csv_path,"topology",rc_label_to_params,['iteration', 'h1','h2'])




def rb_params_to_label(h1,h2,qb):
    widths = [32,h1,h2,1]
    return '_'.join(map(lambda x: f"{x:03d}",widths)) + f"_{qb:02d}b"

def rb_label_to_params(label):
    split_label = label.split("_")
    h1 = int(split_label[1])
    h2 = int(split_label[2])
    qb = int(split_label[4])
    return {"h1":h1,"h2":h2, qb:"qb"}

def rb_setup(csv_path):

    data = pd.read_csv(csv_path).set_index("idx")

    x1 = ChoiceParameter(name="h1", values=[10,20,40,80,160,320,640], parameter_type=ParameterType.INT, is_ordered=True, sort_values=True)
    x2 = ChoiceParameter(name="h2", values=[10,20,40,80,160,320,640], parameter_type=ParameterType.INT, is_ordered=True, sort_values=True)
    qb = ChoiceParameter(name="qb", values=[8,16,32], parameter_type=ParameterType.INT, is_ordered=True, sort_values=True)

    parameters=[x1, x2, qb]

    class MetricA(NoisyFunctionMetric):
        def f(self, x: np.ndarray) -> float:
            return float(data.loc[rb_params_to_label(*x),"model_bits"])

    class MetricB(NoisyFunctionMetric):
        def f(self, x: np.ndarray) -> float:
            return float(data.loc[rb_params_to_label(*x),"data_bits/data_samples"])

    metric_a = MetricA("model_bits", ["h1", "h2", "qb"], noise_sd=0.0, lower_is_better=True)
    metric_b = MetricB("data_bits/data_samples", ["h1", "h2", "qb"], noise_sd=0.0, lower_is_better=True)

    metrics = [metric_a,metric_b]

    return build_ax_config_objects_mohpo(parameters,metrics,data,rb_label_to_params)

def rb_read_glch_data(glch_csv_path):
    return read_sorted_glch_data(glch_csv_path,"idx",rb_label_to_params,['iteration', 'qb', 'h1','h2'])