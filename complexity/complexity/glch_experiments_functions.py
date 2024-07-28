import os
import pandas as pd
import numpy as np
from glch import GLCHGiftWrapping,GLCHGiftWrappingTieBreak,GLCHAngleRule,GHO2D,GHO
from decimal import Decimal
from glch_utils import save_tree_data, save_hull_data, save_threed_history, save_threed_hull_data
from glch_utils import save_history



def build_glch_tree(
    data,possible_values,x_axis,y_axis,initial_values,to_str_method,constrained,start,scale_x,scale_y,
    debug=True,title=None,debug_folder="debug",select_function="angle_rule"
):
    if select_function == "tie_break":
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
    elif select_function == "angle_rule":
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
        title = f"glch2D_{select_function}_{'constrained' if constrained else 'unconstrained'}_{title}"

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
        title = f"glch1D_weights_{'_'.join(['{:.0e}'.format(w) for w in weights])}_{title}"

        if len(axes)==2:
            r,tree_str = build_gho_tree(data,possible_values,axes,initial_values,to_str_method,constrained,weights,
                debug=debug,title=title,debug_folder=debug_folder,version="2D")
            save_tree_data(data,r,axes[0],axes[1],axes_ranges[0],axes_ranges[1],title,
                x_in_log_scale=x_in_log_scale,x_alias=axes_aliases[0],y_alias=axes_aliases[1],fldr=fldr,tree_str=tree_str)
            
            save_history(data,tree_str,title,fldr=fldr)
        else:
            r,tree_str = build_gho_tree(data,possible_values,axes,initial_values,to_str_method,constrained,weights,
                debug=debug,title=title,debug_folder=debug_folder,version="multidimensional")
            
            save_history(data,tree_str,title,fldr=fldr)
    else:
        ValueError(algo)

    return r,tree_str


def glch_rate_vs_energy(
        csv_path,x_axis,y_axis,title,
        x_range=None,y_range=None,
        x_in_log_scale=False,
        x_alias=None,y_alias=None,
        algo="glch",
        constrained=True,
        lmbda=1,
        fldr="glch_results",
        debug_folder="debug",
        debug=True,
        select_function="angle_rule"
    ):

    data = pd.read_csv(csv_path)

    data = data.set_index("topology")

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
        select_function="angle_rule"
    ):

    data = pd.read_csv(csv_path)

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
        select_function="angle_rule"
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
    
    formatted_lambdas = \
        "" if len(lambdas)==0 else "lambdas_" + "-".join([lambdas[i] for i in np.argsort(list(map(float,lambdas)))])+"_"

    exp_id = f'{formatted_lambdas}{"_vs_".join(axes)}_start_{start}'

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
    select_function="angle_rule",
    axes_aliases=None
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
            axes_aliases=axes_aliases,
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

    exp_id = f'glch3D_{select_function}_{"constrained" if constrained else "unconstrained"}_{"_vs_".join(axes)}_start_{start}'

    # save_threed_hull_data(data,rs,axes,complexity_axis,exp_id,fldr=fldr)

    save_threed_history(data,tree_strs,exp_id,fldr=fldr)



