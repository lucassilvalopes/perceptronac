
from scipy import interpolate

def estimate_joules(data,power_draw):

    x = power_draw[:,0]
    y = power_draw[:,1]
    f = interpolate.interp1d(x, y,fill_value="extrapolate")

    # baseline = np.min(y) # this method to estimate the baseline does not always work. 
    baseline = 0 # Maybe I should just leave every measurement biased by the same amount as they are
    constant_power_estimate = (f((data["start_time"].values + data["end_time"].values)/2)-baseline)
    duration = (data["end_time"].values - data["start_time"].values)
    joules = constant_power_estimate * duration
    return joules