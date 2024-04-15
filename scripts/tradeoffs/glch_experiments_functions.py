import os
import pandas as pd
import numpy as np
from perceptronac.power_consumption import estimate_joules, get_n_pixels
from perceptronac.power_consumption import group_energy_measurements
from glch import GLCHGiftWrapping,GLCHGiftWrappingTieBreak,GLCHAngleRule,GHO2D,GHO
from decimal import Decimal
from glch_utils import save_tree_data, save_hull_data, save_threed_history, save_threed_hull_data, save_optimal_point
from glch_utils import save_history



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


def save_glch_data(
    algo,
    data,possible_values,axes,initial_values,to_str_method,constrained,weights,start,
    axes_scales,
    debug,title,debug_folder,select_function,
    x_in_log_scale,axes_ranges,axes_aliases,fldr):

    if algo == "glch":
        title = f"{title}_{select_function}"

        r,tree_str = build_glch_tree(
            data,possible_values,axes[0],axes[1],initial_values,to_str_method,constrained,start,
            scale_x=axes_scales[0],scale_y=axes_scales[1],debug=debug,title=title,debug_folder=debug_folder,
            select_function=select_function)
        save_tree_data(data,r,axes[0],axes[1],axes_ranges[0],axes_ranges[1],title,
            x_in_log_scale=x_in_log_scale,x_alias=axes_aliases[0],y_alias=axes_aliases[1],fldr=fldr,tree_str=tree_str)
        save_hull_data(data,r,axes[0],axes[1],axes_ranges[0],axes_ranges[1],title,
            x_in_log_scale=x_in_log_scale,x_alias=axes_aliases[0],y_alias=axes_aliases[1],fldr=fldr)
        save_history(data,tree_str,title,fldr=fldr)
    elif algo == "gho":
        title = f"{title}_weights_{'_'.join(['{:.0e}'.format(w) for w in weights])}"

        if len(axes)==2:
            r,tree_str = build_gho_tree(data,possible_values,axes,initial_values,to_str_method,constrained,weights,
                debug=debug,title=title,debug_folder=debug_folder,version="2D")
            save_tree_data(data,r,axes[0],axes[1],axes_ranges[0],axes_ranges[1],title,
                x_in_log_scale=x_in_log_scale,x_alias=axes_aliases[0],y_alias=axes_aliases[1],fldr=fldr,tree_str=tree_str)
            # save_optimal_point(data,r,axes,weights,tree_str,title,fldr=fldr)
            save_history(data,tree_str,title,fldr=fldr)
        else:
            r,tree_str = build_gho_tree(data,possible_values,axes,initial_values,to_str_method,constrained,weights,
                debug=debug,title=title,debug_folder=debug_folder,version="multidimensional")
            # save_optimal_point(data,r,axes,weights,tree_str,title,fldr=fldr)
            save_history(data,tree_str,title,fldr=fldr)
    else:
        ValueError(algo)

    return r,tree_str


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
        debug=True,
        select_function="corrected_angle_rule"
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

    start="left"
    weights = [1,lmbda]
    axes = [x_axis,y_axis]
    axes_ranges=[x_range,y_range]
    axes_aliases=[x_alias,y_alias]
    axes_scales = [scale_x,scale_y]
    save_glch_data(
        algo,
        data,possible_values,axes,initial_values,to_str_method,constrained,weights,start,
        axes_scales,
        debug,title,debug_folder,select_function,
        x_in_log_scale,axes_ranges,axes_aliases,fldr)


def glch_rate_vs_time(*args,**kwargs):
    glch_rate_vs_energy(*args,**kwargs)


def glch_rate_vs_params(*args,**kwargs):
    glch_rate_vs_energy(*args,**kwargs)


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
        debug=True,
        select_function="corrected_angle_rule"
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

    start="left"
    weights = [1,lmbda]
    title='model_bits_vs_data_bits'
    axes = [x_axis,y_axis]
    axes_ranges=[x_range,y_range]
    axes_aliases=[x_alias,y_alias]
    axes_scales = [scale_x,scale_y]
    save_glch_data(
        algo,
        data,possible_values,axes,initial_values,to_str_method,constrained,weights,start,
        axes_scales,
        debug,title,debug_folder,select_function,
        x_in_log_scale,axes_ranges,axes_aliases,fldr)




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
        debug=True,
        select_function="corrected_angle_rule"
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

    x_in_log_scale = False
    title = exp_id
    axes_scales = [scale_x,scale_y]
    r,tree_str = save_glch_data(
        algo,
        data,possible_values,axes,initial_values,to_str_method,constrained,weights,start,
        axes_scales,
        debug,title,debug_folder,select_function,
        x_in_log_scale,axes_ranges,axes_aliases,fldr)
    
    return r,tree_str


def glch3d_rdc(
    csv_path,
    complexity_axis="params",
    constrained=True,
    start="left",
    lambdas=["5e-3", "1e-2", "2e-2"],
    fldr="glch_results",
    debug_folder="debug",
    debug=True,
    select_function="corrected_angle_rule"
):

    rs = []
    tree_strs = []
    for lmbda in lambdas:
        r,tree_str = glch_rate_vs_dist(
            csv_path,
            [complexity_axis,"loss"],
            algo="glch",
            constrained=constrained,
            weights=None,
            start=start,
            lambdas= [lmbda],
            axes_ranges=None,
            axes_aliases=None,
            fldr=fldr,
            debug_folder=debug_folder,
            debug=debug,
            select_function=select_function
        )
        tree_strs.append(tree_str)
        rs.append(r)

    data = pd.read_csv(csv_path)
    data = data.set_index("labels")

    axes = ["bpp_loss","mse_loss",complexity_axis]

    exp_id = f'{"_vs_".join(axes)}_start_{start}_{select_function}'

    save_threed_hull_data(data,rs,axes,complexity_axis,exp_id,fldr=fldr)

    save_threed_history(data,tree_strs,exp_id,fldr=fldr)



