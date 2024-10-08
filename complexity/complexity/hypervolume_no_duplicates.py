#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np


from complexity.ax_utils import build_optimization_config_mohpo, get_hv_from_df
from ax.core.search_space import SearchSpace


# In[2]:


def get_glch_max_complexity_history(glch_data,complexity_axis):

    # glch_data = glch_data.drop_duplicates(subset=[glch_data.index.name])

    glch_data = glch_data[~glch_data.index.duplicated(keep='first')]
    
    max_complexity_history = []

    for i in range(glch_data.shape[0]):

        curr_data = glch_data.iloc[:i+1,:]

        curr_max = curr_data[complexity_axis].max()
        
        max_complexity_history.append(curr_max)
    
    return max_complexity_history


# In[3]:


def equalize_lists_sizes_in_dict(dict_of_lists):

    max_len = max([len(dict_of_lists) for dict_of_lists in dict_of_lists.values()])
    
    res = {k:v+((max_len-len(v))*[None]) for k,v in dict_of_lists.items()}

    return res


# In[4]:


def getting_the_thresholds(glch_csv_paths,read_glch_data_func,metrics):
    glch_max_c_lists = dict()
    all_glch_data = dict()
    for lbl,glch_csv_path in glch_csv_paths.items():
        glch_data = read_glch_data_func(glch_csv_path)
        all_glch_data[lbl] = glch_data
        glch_max_c_lists[lbl] = get_glch_max_complexity_history(glch_data,metrics[0].name)
    
    glch_max_c_lists = equalize_lists_sizes_in_dict(glch_max_c_lists)
    
    thresholds = pd.DataFrame(glch_max_c_lists).min(axis=1).values.tolist()

    return thresholds,all_glch_data


# In[5]:


def hv_vs_n_trained_networks(
    ax_results_folder,label_to_params_func,params_to_label_func,
    parameters,metrics,data,search_space,prefix,thresholds,all_glch_data,up_to_complexity=False):
    
    all_hvs = {k:[] for k in all_glch_data.keys()}
    all_hvs["max_hv"] = []
    all_hvs = {**all_hvs, "sobol":[],"ehvi":[],"parego":[]}
    all_hvs["complexity"] = []

    if not up_to_complexity:
        ref_point = data[[metric.name for metric in metrics]].max().values * 1.1
        optimization_config = build_optimization_config_mohpo(metrics,ref_point)

        ith_max_hv = get_hv_from_df(search_space,optimization_config,data,label_to_params_func)
    
    for i,th in enumerate(thresholds,1):

        all_hvs["complexity"].append(th)

        if up_to_complexity:
            filt_data = data[data[metrics[0].name] <= th]

            # ref_point = filt_data[[metric.name for metric in metrics]].max().values * 1.1
            ref_point = data[[metric.name for metric in metrics]].max().values * 1.1
            optimization_config = build_optimization_config_mohpo(metrics,ref_point)

            ith_max_hv = get_hv_from_df(search_space,optimization_config,filt_data,label_to_params_func)
    
        all_hvs["max_hv"].append( ith_max_hv )
        
        for lbl,glch_data in all_glch_data.items():

            # filt_glch_data = glch_data.drop_duplicates(subset=[data.index.name])

            filt_glch_data = glch_data[~glch_data.index.duplicated(keep='first')]

            if filt_glch_data.shape[0] < i:
                continue
    
            filt_glch_data = filt_glch_data.iloc[:i,:]

            if up_to_complexity:
                filt_glch_data = filt_glch_data[filt_glch_data[metrics[0].name] <= th]
            
            all_hvs[lbl].append( get_hv_from_df(search_space,optimization_config,filt_glch_data,label_to_params_func) )
    
    
    
        for method in ["sobol","ehvi","parego"]:
            method_hvs = []
            for f in os.listdir(ax_results_folder):
                if f.endswith(".csv") and (prefix in f):
                    ax_df = pd.read_csv(os.path.join(ax_results_folder,f))
                
                    tmp_df = ax_df[[c for c in ax_df.columns if f"{method}_param_" in c]].copy()
                    tmp_df.columns = [c.replace(f"{method}_param_","") for c in tmp_df.columns]
                    tmp_df.loc[:,data.index.name] = tmp_df.apply(lambda x: params_to_label_func(*x.values),axis=1)
                    tmp_df = tmp_df.reset_index(names="iteration").merge(
                        data,left_on=data.index.name,right_on=data.index.name).sort_values(by="iteration") 
                        # .set_index(data.index.name).sort_values(by="iteration")

                    filt_tmp_df = tmp_df.drop_duplicates(subset=[data.index.name])

                    if filt_tmp_df.shape[0] < i:
                        continue
                    
                    filt_tmp_df = filt_tmp_df.iloc[:i,:]

                    if up_to_complexity:
                        filt_tmp_df = filt_tmp_df[filt_tmp_df[metrics[0].name] <= th]

                    # if len(set(filt_tmp_df.index)) != len(filt_tmp_df.index):
                    #     raise ValueError(f"{f} {len(set(filt_tmp_df.index))} {len(filt_tmp_df.index)}")
    
                    if filt_tmp_df.shape[0] == 0:
                        method_hvs.append(0)
                    else:
                        method_hvs.append( get_hv_from_df(
                            search_space,optimization_config,filt_tmp_df.set_index(data.index.name),label_to_params_func) )

            if len(method_hvs) > 0:
                all_hvs[method].append( sum(method_hvs)/len(method_hvs) )

    return all_hvs


# In[6]:


def adjust_ax_glch_comparison_mohpo(
    ax_results_folder,data_csv_path,setup_func,glch_csv_paths,read_glch_data_func,label_to_params_func,params_to_label_func,
    up_to_complexity=False
):

    parameters,metrics,data = setup_func(data_csv_path)
    
    search_space = SearchSpace(parameters=parameters)
    
    prefix = f"{'_'.join([metric.name for metric in metrics]).replace('/','_over_')}_ax_methods_seed"
    
    thresholds,all_glch_data = getting_the_thresholds(glch_csv_paths,read_glch_data_func,metrics)
    
    all_hvs = hv_vs_n_trained_networks(
        ax_results_folder,label_to_params_func,params_to_label_func,
        parameters,metrics,data,search_space,prefix,thresholds,all_glch_data,up_to_complexity=up_to_complexity)
    
    all_hvs_df = pd.DataFrame(equalize_lists_sizes_in_dict(all_hvs))
    
    all_hvs_df = all_hvs_df.assign(iters=np.arange(1,all_hvs_df.shape[0]+1) ).set_index("iters")
    
    all_hvs_df.to_csv(f"{prefix.replace('_seed','_avgs')}_adjusted{('_up_to_complexity' if up_to_complexity else '')}.csv")

    return all_hvs_df







