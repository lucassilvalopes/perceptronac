import os
import pandas as pd
import numpy as np
from perceptronac.power_consumption import estimate_joules, get_n_pixels
from perceptronac.power_consumption import group_energy_measurements
from glch import GLCHGiftWrapping,GLCHGiftWrappingTieBreak,GLCHAngleRule,GHO2D,GHO
from decimal import Decimal
from glch_utils import save_tree_data, save_hull_data, save_trees_data, save_hulls_data, save_optimal_point



def build_glch_tree(
    data,possible_values,x_axis,y_axis,initial_values,to_str_method,constrained,start,scale_x,scale_y,
    debug=True,title=None,debug_folder="debug",select_function="corrected_angle_rule"
):
    if select_function == "gift_wrapping_tie_break":
        glch_alg = GLCHGiftWrappingTieBreak(
            data,possible_values,[x_axis,y_axis],initial_values,to_str_method,constrained,[scale_x,scale_y],
            debug,title,debug_folder
        )
        r = glch_alg.build_tree()
        return r, glch_alg.get_tree_str()
    elif select_function == "gift_wrapping":
        glch_alg = GLCHGiftWrapping(
            data,possible_values,[x_axis,y_axis],initial_values,to_str_method,constrained,start,
            debug,title,debug_folder
        )
        r = glch_alg.build_tree()
        return r, glch_alg.get_tree_str()
    elif select_function == "corrected_angle_rule":
        glch_alg = GLCHAngleRule(
            data,possible_values,[x_axis,y_axis],initial_values,to_str_method,constrained,start,
            debug,title,debug_folder
        )
        r = glch_alg.build_tree()
        return r, glch_alg.get_tree_str()
    else:
        ValueError(select_function)


def build_gho_tree(
    data,possible_values,axes,initial_values,to_str_method,constrained,weights,
    debug=True,title=None,debug_folder="debug",version="multidimensional"
):
    if version== "2D":
        gho_alg = GHO2D(
            data,possible_values,axes,initial_values,to_str_method,constrained,
            weights,debug,title,debug_folder
        )
        r = gho_alg.build_tree()
        return r, gho_alg.get_tree_str()
    elif version== "multidimensional":
        gho_alg = GHO(
            data,possible_values,axes,initial_values,to_str_method,constrained,
            weights
        )
        r = gho_alg.build_tree()
        return r, gho_alg.get_tree_str()
    else:
        ValueError(version)


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
        debug_folder="debug",
        debug=True
    ):

    data = get_energy_data(csv_path,remove_noise)

    scale_x = data.loc[["032_010_010_001","032_640_640_001"],x_axis].max() - data.loc[["032_010_010_001","032_640_640_001"],x_axis].min()
    scale_y = data.loc[["032_010_010_001","032_640_640_001"],y_axis].max() - data.loc[["032_010_010_001","032_640_640_001"],y_axis].min()

    possible_values = {
        "h1": [10,20,40,80,160,320,640],
        "h2": [10,20,40,80,160,320,640]
    }

    initial_values = {"h1":10,"h2":10}

    def to_str_method(params):
        widths = [32,params["h1"],params["h2"],1]
        return '_'.join(map(lambda x: f"{x:03d}",widths))

    if algo == "glch":
        r,tree_str = build_glch_tree(data,possible_values,x_axis,y_axis,initial_values,to_str_method,constrained,"left",
            scale_x=scale_x,scale_y=scale_y,debug=debug,title=title,debug_folder=debug_folder)
    elif algo == "gho":
        r,tree_str = build_gho_tree(data,possible_values,[x_axis,y_axis],initial_values,to_str_method,constrained,[1,lmbda],
            debug=debug,title=title,debug_folder=debug_folder,version="2D")
    else:
        ValueError(algo)

    save_tree_data(data,r,x_axis,y_axis,x_range,y_range,title,
        x_in_log_scale=x_in_log_scale,x_alias=x_alias,y_alias=y_alias,fldr=fldr,tree_str=tree_str)
    if algo == "glch":
        save_hull_data(data,r,x_axis,y_axis,x_range,y_range,title,
            x_in_log_scale=x_in_log_scale,x_alias=x_alias,y_alias=y_alias,fldr=fldr)


def glch_rate_vs_time(*args,**kwargs):
    glch_rate_vs_energy(*args,**kwargs)


def glch_rate_vs_params(*args,**kwargs):
    glch_rate_vs_energy(*args,**kwargs)


def glch_rate_vs_dist(
        csv_path,
        axes,
        algo="glch",
        constrained=True,
        weights=None,
        start="left",
        lambdas=[],
        axes_ranges=None,
        axes_aliases=None,
        fldr="glch_results",
        debug_folder="debug",
        debug=True
    ):

    data = pd.read_csv(csv_path)

    if len(lambdas) == 0:
        data = data.set_index("labels")
    else:
        data = data[data["labels"].apply(lambda x: any([(lmbd in x) for lmbd in lambdas]) )].set_index("labels")

    x_axis = axes[0]
    y_axis = axes[1]

    simplest = "D3L{}N32M32".format("5e-3" if len(lambdas) == 0 else lambdas[np.argmin(list(map(float,lambdas)))])
    most_complex = "D4L{}N224M320".format("2e-2" if len(lambdas) == 0 else lambdas[np.argmax(list(map(float,lambdas)))])

    scale_x = data.loc[[simplest,most_complex],x_axis].max() - data.loc[[simplest,most_complex],x_axis].min()
    scale_y = data.loc[[simplest,most_complex],y_axis].max() - data.loc[[simplest,most_complex],y_axis].min()

    possible_values = {
        "D": [3,4],
        "L": ["5e-3", "1e-2", "2e-2"] if len(lambdas) == 0 else lambdas,
        "N": [32, 64, 96, 128, 160, 192, 224],
        "M": [32, 64, 96, 128, 160, 192, 224, 256, 288, 320]
    }

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
    
    formatted_lambdas = "" if len(lambdas)==0 else "_" + "-".join([lambdas[i] for i in np.argsort(list(map(float,lambdas)))])

    exp_id = f'{"_vs_".join(axes)}_start_{start}{formatted_lambdas}'

    if weights is None:
        weights = [1 for _ in range(len(axes))]
    if axes_ranges is None:
        axes_ranges = [None for _ in range(len(axes))]
    if axes_aliases is None:
        axes_aliases = [None for _ in range(len(axes))]

    if algo == "glch":
        r,tree_str = build_glch_tree(data,possible_values,axes[0],axes[1],initial_values,to_str_method,constrained,start,
            scale_x=scale_x,scale_y=scale_y,debug=debug,title=None,debug_folder=debug_folder)
        save_tree_data(data,r,axes[0],axes[1],axes_ranges[0],axes_ranges[1],exp_id,
            x_alias=axes_aliases[0],y_alias=axes_aliases[1],fldr=fldr,tree_str=tree_str)
        save_hull_data(data,r,axes[0],axes[1],axes_ranges[0],axes_ranges[1],exp_id,
            x_alias=axes_aliases[0],y_alias=axes_aliases[1],fldr=fldr)
    elif algo == "gho":
        if len(axes)==2:
            r,tree_str = build_gho_tree(data,possible_values,axes,initial_values,to_str_method,constrained,weights,
                debug=debug,title=None,debug_folder=debug_folder,version="2D")
            save_tree_data(data,r,axes[0],axes[1],axes_ranges[0],axes_ranges[1],exp_id,
                x_alias=axes_aliases[0],y_alias=axes_aliases[1],fldr=fldr,tree_str=tree_str)
            save_optimal_point(data,r,axes,weights,tree_str,exp_id,fldr=fldr)
        else:
            r,tree_str = build_gho_tree(data,possible_values,axes,initial_values,to_str_method,constrained,weights,
                debug=debug,title=None,debug_folder=debug_folder,version="multidimensional")
            save_optimal_point(data,r,axes,weights,tree_str,exp_id,fldr=fldr)
    else:
        ValueError(algo)


def glch_rate_vs_dist_2(
    csv_path,x_axis,y_axis,x_range=None,y_range=None,start="left",constrained=True,fldr="glch_results",debug_folder="debug",debug=True):
    """only for glch algo"""

    data = pd.read_csv(csv_path).set_index("labels")

    scale_x = data.loc[["D3L5e-3N32M32","D4L2e-2N224M320"],x_axis].max() - data.loc[["D3L5e-3N32M32","D4L2e-2N224M320"],x_axis].min()
    scale_y = data.loc[["D3L5e-3N32M32","D4L2e-2N224M320"],y_axis].max() - data.loc[["D3L5e-3N32M32","D4L2e-2N224M320"],y_axis].min()

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
    tree_strs = []
    for i,L in enumerate(brute_dict["L"]):

        to_str_method = to_str_method_factory({"L":L})

        current_data = data.iloc[[i for i,lbl in enumerate(data.index) if f"L{L}" in lbl],:]
        r,tree_str = build_glch_tree(current_data,greedy_dict,x_axis,y_axis,initial_state,to_str_method,constrained,start,
            scale_x=scale_x,scale_y=scale_y,debug=debug,title=None,debug_folder=debug_folder)
        
        tree_strs.append(tree_str)
        rs.append(r)

    save_trees_data(data,rs,brute_dict["L"],x_axis,y_axis,x_range,y_range,exp_id,fldr=fldr,tree_strs=tree_strs)
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
        debug_folder="debug",
        debug=True
    ):

    data = pd.read_csv(csv_path)

    data["model_bits"] = data["model_bits/data_samples"] * data["data_samples"]

    data['idx'] = data.apply(lambda x: f"{x.topology}_{x.quantization_bits:02d}b", axis=1)

    data = data.set_index("idx")

    scale_x = data.loc[["032_010_010_001_08b","032_640_640_001_32b"],x_axis].max() - data.loc[["032_010_010_001_08b","032_640_640_001_32b"],x_axis].min()
    scale_y = data.loc[["032_010_010_001_08b","032_640_640_001_32b"],y_axis].max() - data.loc[["032_010_010_001_08b","032_640_640_001_32b"],y_axis].min()

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
        r,tree_str = build_glch_tree(
            data,possible_values,x_axis,y_axis,initial_values,to_str_method,constrained,"left",
            scale_x=scale_x,scale_y=scale_y,debug=debug,title=None,debug_folder=debug_folder)
    elif algo == "gho":
        r,tree_str = build_gho_tree(data,possible_values,[x_axis,y_axis],initial_values,to_str_method,constrained,[1,lmbda],
            debug=debug,title=None,debug_folder=debug_folder,version="2D")
    else:
        ValueError(algo)

    save_tree_data(data,r,x_axis,y_axis,x_range,y_range,'model_bits_vs_data_bits',
        x_in_log_scale=x_in_log_scale,x_alias=x_alias,y_alias=y_alias,fldr=fldr,tree_str=tree_str)

    if algo == "glch":
        save_hull_data(data,r,x_axis,y_axis,x_range,y_range,'model_bits_vs_data_bits',
            x_in_log_scale=x_in_log_scale,x_alias=x_alias,y_alias=y_alias,fldr=fldr)
