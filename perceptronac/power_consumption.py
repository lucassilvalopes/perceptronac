
from scipy import interpolate
import numpy as np
import pandas as pd


def group_energy_measurements(data):

    data = data.groupby(["topology","quantization_bits"]).apply(
        lambda x: pd.Series({
            "data_bits/data_samples":x["data_bits/data_samples"].iloc[0],
            "(data_bits+model_bits)/data_samples":x["(data_bits+model_bits)/data_samples"].iloc[0],
            "model_bits/data_samples":x["model_bits/data_samples"].iloc[0],
            "data_samples":x["data_samples"].iloc[0],
            "topology": x["topology"].iloc[0], 
            "params": x["params"].iloc[0],
            "quantization_bits": x["quantization_bits"].iloc[0],
            "joules": x["joules"].mean(),
            "joules_std": x["joules"].std()
        },index=[
            "data_bits/data_samples",
            "(data_bits+model_bits)/data_samples",
            "model_bits/data_samples",
            "data_samples",
            "topology",
            "params",
            "quantization_bits",
            "joules",
            "joules_std"
        ]))
    
    static_data = data.loc[data["quantization_bits"]==32,:].drop(
        ["model_bits/data_samples","(data_bits+model_bits)/data_samples", "quantization_bits"], axis=1)

    static_data = static_data.sort_values("data_bits/data_samples")

    return static_data


def estimate_joules(data,power_draw):
    return estimate_joules_integral(data,power_draw)


def estimate_joules_integral(data,power_draw):
    slices = []
    for i,row in data.iterrows():
        slices.append(
            power_draw[:,1][np.logical_and(power_draw[:,0] > row["start_time"],power_draw[:,0] < row["end_time"])]
        )

    # the points are distant 1s from each other, then the integral can be approximated by simply the sum
    joules = [np.sum(lst) for lst in slices]
    
    # mean power and std
    # power_mean = [np.mean(lst) for lst in slices] 
    # power_std = [np.std(lst) for lst in slices]

    return joules

def estimate_joules_constant_power_linear_interpolation(data,power_draw):

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    # https://stackoverflow.com/questions/45429831/valueerror-a-value-in-x-new-is-above-the-interpolation-range-what-other-re

    x = power_draw[:,0]
    y = power_draw[:,1]
    f = interpolate.interp1d(x, y,fill_value="extrapolate")

    # baseline = np.min(y) # this method to estimate the baseline does not always work. 
    baseline = 0 # Maybe I should just leave every measurement biased by the same amount as they are
    constant_power_estimate = (f((data["start_time"].values + data["end_time"].values)/2)-baseline)
    duration = (data["end_time"].values - data["start_time"].values)
    joules = constant_power_estimate * duration
    return joules