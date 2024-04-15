
from scipy import interpolate
import ast
from PIL import Image
import numpy as np
import pandas as pd
from decimal import Decimal


def get_n_pixels(conf_path):
    conf = pd.read_csv(conf_path,index_col=0,header=0)
    n_pixels = 0
    for im_path in ast.literal_eval(conf.loc['validation_set','value']):
        im = Image.open(im_path)
        n_pixels += (im.size[0]*im.size[1])
    return n_pixels


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
            "joules_std": x["joules"].std(),
            # "start_time": x["start_time"].mean(),
            # "end_time": x["end_time"].mean(),
            "time": x["end_time"].mean() - x["start_time"].mean(),
            "time_std": np.sqrt(x["end_time"].var() + x["start_time"].var() - 2*x[["end_time","start_time"]].cov().iloc[0,1])
        },index=[
            "data_bits/data_samples",
            "(data_bits+model_bits)/data_samples",
            "model_bits/data_samples",
            "data_samples",
            "topology",
            "params",
            "quantization_bits",
            "joules",
            "joules_std",
            # "start_time",
            # "end_time",
            "time",
            "time_std"
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



def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1


def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()


def limit_significant_digits(value,last_significant_digit_position):
    factor = 10**last_significant_digit_position
    return np.round(value/factor) * factor


def limit_energy_significant_digits(data,x_axis):
    """
    https://stackoverflow.com/questions/45332056/decompose-a-float-into-mantissa-and-exponent-in-base-10-without-strings
    """

    mean_col=x_axis
    std_col=f"{x_axis}_std"

    last_significant_digit_position = fexp(data[std_col].max())

    data[[mean_col,std_col]] = data[[mean_col,std_col]].apply(lambda x: pd.Series({
        mean_col:limit_significant_digits(x[mean_col],last_significant_digit_position),
        std_col:limit_significant_digits(x[std_col],last_significant_digit_position)
        # mean_col:limit_significant_digits(x[mean_col],fexp(x[std_col])),
        # std_col:limit_significant_digits(x[std_col],fexp(x[std_col]))
    },index=[mean_col,std_col]), axis=1)
    return data


def get_energy_data(csv_path,remove_noise):

    data = pd.read_csv(csv_path)

    csv_path_2 = csv_path.replace("raw_values","power_draw")

    power_draw = np.loadtxt(csv_path_2)

    power_draw[:,1] = power_draw[:,1] - 16 # np.min(power_draw[:,1])

    joules = estimate_joules(data,power_draw)

    data["joules"] = joules

    data = group_energy_measurements(data).set_index("topology")

    csv_path_3 = csv_path.replace("raw_values","conf")

    n_pixels = get_n_pixels(csv_path_3)

    data["joules_per_pixel"] = data["joules"] / n_pixels

    data["joules_per_pixel_std"] = data["joules_std"] / n_pixels

    data["micro_joules_per_pixel"] = data["joules_per_pixel"] / 1e-6

    data["micro_joules_per_pixel_std"] = data["joules_per_pixel_std"] / 1e-6
    
    if remove_noise:

        limit_energy_significant_digits(data,"time")
        limit_energy_significant_digits(data,"joules")
        limit_energy_significant_digits(data,"joules_per_pixel")
        limit_energy_significant_digits(data,"micro_joules_per_pixel")

    return data


if __name__ == "__main__":

    get_energy_data(
        "/home/lucas/Documents/perceptronac/results/exp_1676160746/exp_1676160746_raw_values.csv",False
    ).to_csv(
        "/home/lucas/Documents/perceptronac/scripts/tradeoffs/rate-noisy-joules-time-params_hx-10-20-40-80-160-320-640.csv"
    )


    