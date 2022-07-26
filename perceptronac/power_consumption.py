
from scipy import interpolate
import numpy as np


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