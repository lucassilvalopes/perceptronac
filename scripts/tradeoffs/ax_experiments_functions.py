
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

from ax_utils import build_ax_config_objects, read_sorted_glch_data, ax_glch_comparison




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

    return build_ax_config_objects(parameters,metrics,data,rdc_label_to_params)



def rdc_read_glch_data(glch_csv_path):
    return read_sorted_glch_data(glch_csv_path,"labels",rdc_label_to_params,['iteration', 'D','N','M','L'])



