import os
import pandas as pd
import numpy as np
from perceptronac.power_consumption import estimate_joules, get_n_pixels
from perceptronac.power_consumption import group_energy_measurements
from glch_BandD import GLCH
from decimal import Decimal
from glch_utils import save_tree_data, save_hull_data, save_trees_data, save_hulls_data



def build_glch_tree(
    data,possible_values,x_axis,y_axis,initial_values,to_str_method,start="left",debug=True,title=None,
    constrained=True, debug_folder="debug"
):
    return GLCH(
        data,possible_values,x_axis,y_axis,initial_values,to_str_method,start,debug,title,
        constrained, "corrected_angle_rule", 1, debug_folder
    ).build_tree()


def build_gho_tree(
    data,possible_values,x_axis,y_axis,initial_values,to_str_method,start="left",debug=True,title=None,
    constrained=True,lmbda=1, debug_folder="debug"
):
    return GLCH(
        data,possible_values,x_axis,y_axis,initial_values,to_str_method,start,debug,title,
        constrained, "point", lmbda, debug_folder
    ).build_tree()


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


def glch_rate_vs_energy(
        csv_path,x_axis,y_axis,title,
        x_range=None,y_range=None,
        x_in_log_scale=False,remove_noise=True,
        x_alias=None,y_alias=None,
        algo="glch",
        constrained=True,
        lmbda=1,
        fldr="glch_results",
        debug_folder="debug"
    ):

    data = get_energy_data(csv_path,remove_noise)

    possible_values = {
        "h1": [10,20,40,80,160,320,640],
        "h2": [10,20,40,80,160,320,640]
    }

    initial_values = {"h1":10,"h2":10}

    def to_str_method(params):
        widths = [32,params["h1"],params["h2"],1]
        return '_'.join(map(lambda x: f"{x:03d}",widths))

    if algo == "glch":
        r = build_glch_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,title=title,
            constrained=constrained,debug_folder=debug_folder)
    elif algo == "gho":
        r = build_gho_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,title=title,
            constrained=constrained,lmbda=lmbda,debug_folder=debug_folder)
    else:
        ValueError(algo)

    save_tree_data(data,r,x_axis,y_axis,x_range,y_range,title,
        x_in_log_scale=x_in_log_scale,x_alias=x_alias,y_alias=y_alias,fldr=fldr)
    if algo == "glch":
        save_hull_data(data,r,x_axis,y_axis,x_range,y_range,title,
            x_in_log_scale=x_in_log_scale,x_alias=x_alias,y_alias=y_alias,fldr=fldr)


def glch_rate_vs_time(*args,**kwargs):
    glch_rate_vs_energy(*args,**kwargs)


def glch_rate_vs_params(*args,**kwargs):
    glch_rate_vs_energy(*args,**kwargs)


def glch_rate_vs_dist(
        csv_path,x_axis,y_axis,
        x_range=None,y_range=None,
        start="left",
        x_alias=None,y_alias=None,
        lambdas=[],
        algo="glch",
        constrained=True,
        lmbda=1,
        fldr="glch_results",
        debug_folder="debug"
    ):

    data = pd.read_csv(csv_path)

    if len(lambdas) == 0:
        data = data.set_index("labels")
    else:
        data = data[data["labels"].apply(lambda x: any([(lmbd in x) for lmbd in lambdas]) )].set_index("labels")

    possible_values = {
        "D": [3,4],
        "L": ["5e-3", "1e-2", "2e-2"] if len(lambdas) == 0 else lambdas,
        "N": [32, 64, 96, 128, 160, 192, 224],
        "M": [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
    }

    # x_axis = "bpp_loss"
    # y_axis = "mse_loss"

    if start == "right":
        possible_values = {k:v[::-1] for k,v in possible_values.items()}

    initial_values = {
        "D":possible_values["D"][0], 
        "L":possible_values["L"][0], 
        "N":possible_values["N"][0],
        "M":possible_values["M"][0]
    }

    def to_str_method(params):
        return f"D{params['D']}L{params['L']}N{params['N']}M{params['M']}"
    
    if algo == "glch":
        if start == "right":
            r = build_glch_tree(data,possible_values,y_axis,x_axis,initial_values,to_str_method,
                constrained=constrained,debug_folder=debug_folder)
        else:
            r = build_glch_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,
                constrained=constrained,debug_folder=debug_folder)
    elif algo == "gho":
        r = build_gho_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,
            constrained=constrained,lmbda=lmbda,debug_folder=debug_folder)
    else:
        ValueError(algo)

    formatted_lambdas = "" if len(lambdas)==0 else "_" + "-".join([lambdas[i] for i in np.argsort(list(map(float,lambdas)))])

    save_tree_data(data,r,x_axis,y_axis,x_range,y_range,f'{x_axis}_vs_{y_axis}_start_{start}{formatted_lambdas}',
        x_alias=x_alias,y_alias=y_alias,fldr=fldr)
    if algo == "glch":
        save_hull_data(data,r,x_axis,y_axis,x_range,y_range,f'{x_axis}_vs_{y_axis}_start_{start}{formatted_lambdas}',
            x_alias=x_alias,y_alias=y_alias,fldr=fldr)


def glch_rate_vs_dist_2(
    csv_path,x_axis,y_axis,x_range=None,y_range=None,start="left",constrained=True,fldr="glch_results",debug_folder="debug"):
    """only for glch algo"""

    data = pd.read_csv(csv_path).set_index("labels")

    brute_dict = {
        "L": ["5e-3", "1e-2", "2e-2"]
    }

    greedy_dict = {
        "D": [3,4],
        "N": [32, 64, 96, 128, 160, 192, 224],
        "M": [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
    }

    if start == "right":
        greedy_dict = {k:v[::-1] for k,v in greedy_dict.items()}

    initial_state = {
        "D":greedy_dict["D"][0],
        "N":greedy_dict["N"][0],
        "M":greedy_dict["M"][0]
    }

    def to_str_method_factory(brute_params):
        def to_str_method(greedy_params):
            return f"D{greedy_params['D']}L{brute_params['L']}N{greedy_params['N']}M{greedy_params['M']}"
        return to_str_method


    brute_keys = "".join(list(brute_dict.keys()))
    greedy_keys = "".join(list(greedy_dict.keys()))
    exp_id = f"{x_axis}_vs_{y_axis}_brute_{brute_keys}_greedy_{greedy_keys}_start_{start}"

    rs = []
    for i,L in enumerate(brute_dict["L"]):

        to_str_method = to_str_method_factory({"L":L})

        current_data = data.iloc[[i for i,lbl in enumerate(data.index) if f"L{L}" in lbl],:]
        if start == "right":
            r = build_glch_tree(current_data,greedy_dict,y_axis,x_axis,initial_state,to_str_method,
                constrained=constrained,debug_folder=debug_folder)
        else:
            r = build_glch_tree(current_data,greedy_dict,x_axis,y_axis,initial_state,to_str_method,
                constrained=constrained,debug_folder=debug_folder)
        
        rs.append(r)

    save_trees_data(data,rs,brute_dict["L"],x_axis,y_axis,x_range,y_range,exp_id,fldr=fldr)
    save_hulls_data(data,rs,brute_dict["L"],x_axis,y_axis,x_range,y_range,exp_id,fldr=fldr)


def glch_model_bits_vs_data_bits(
        csv_path,x_axis,y_axis,
        x_range=None,y_range=None,
        x_in_log_scale=False,
        x_alias=None,y_alias=None,
        algo="glch",
        constrained=True,
        lmbda=1,
        fldr="glch_results",
        debug_folder="debug"
    ):

    data = pd.read_csv(csv_path)

    data["model_bits"] = data["model_bits/data_samples"] * data["data_samples"]

    data['idx'] = data.apply(lambda x: f"{x.topology}_{x.quantization_bits:02d}b", axis=1)

    data = data.set_index("idx")

    possible_values = {
        "h1": [10,20,40,80,160,320,640],
        "h2": [10,20,40,80,160,320,640],
        "qb": [8,16,32]
    }

    # x_axis = "model_bits/data_samples"
    # y_axis = "data_bits/data_samples"

    initial_values = {"h1":10,"h2":10,"qb":8}

    def to_str_method(params):
        widths = [32,params["h1"],params["h2"],1]
        return '_'.join(map(lambda x: f"{x:03d}",widths)) + f"_{params['qb']:02d}b"

    if algo == "glch":
        r = build_glch_tree(
            data,possible_values,x_axis,y_axis,initial_values,to_str_method,
            constrained=constrained,debug_folder=debug_folder)
    elif algo == "gho":
        r = build_gho_tree(
            data,possible_values,x_axis,y_axis,initial_values,to_str_method,
            constrained=constrained,lmbda=lmbda,debug_folder=debug_folder)
    else:
        ValueError(algo)

    save_tree_data(data,r,x_axis,y_axis,x_range,y_range,'model_bits_vs_data_bits',
        x_in_log_scale=x_in_log_scale,x_alias=x_alias,y_alias=y_alias,fldr=fldr)

    if algo == "glch":
        save_hull_data(data,r,x_axis,y_axis,x_range,y_range,'model_bits_vs_data_bits',
            x_in_log_scale=x_in_log_scale,x_alias=x_alias,y_alias=y_alias,fldr=fldr)
