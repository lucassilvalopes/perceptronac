#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from collections import Counter


# In[2]:


def ax_methods_trained_networks(ax_results_folder,prefix):

    stats_dict = dict()

    for method in ["sobol","ehvi","parego"]:

        stats_dict[method] = []

        set_lens = []
        list_lens = []
        for f in os.listdir(ax_results_folder):
            if f.endswith(".csv") and (prefix in f):
                ax_df = pd.read_csv(os.path.join(ax_results_folder,f))
                method_param_cols = [c for c in ax_df.columns if f"{method}_param_" in c]
                tmp_df = ax_df[method_param_cols].copy()

                list_len = tmp_df.shape[0]
                set_len = tmp_df.drop_duplicates(subset=method_param_cols).shape[0]

                list_lens.append(list_len)
                set_lens.append(set_len)

                stats_dict[method].append(set_len)

        print(method,min(set_lens),Counter(set_lens),Counter(list_lens))
    
    return stats_dict


# In[3]:


def ax_methods_df_with_counts(adjusted_data_folder,prefix,stats_dict,up_to_complexity):

    fldr = adjusted_data_folder
    if up_to_complexity:
        fil3 = prefix.replace("_seed","_avgs_adjusted_up_to_complexity.csv")
    else:
        fil3 = prefix.replace("_seed","_avgs_adjusted.csv")
    pth = fldr + fil3

    df = pd.read_csv(pth,index_col=0)

    df[["sobol_count","ehvi_count","parego_count"]] = 0

    for method in ["sobol","ehvi","parego"]:
        for n_unique in stats_dict[method]:
            col = df.columns.tolist().index(f"{method}_count")
            df.iloc[:n_unique,col] = df.iloc[:n_unique,col] + 1

    return df


# In[4]:


def col_idx(df,col):
    return df.columns.tolist().index(col)


# In[5]:


def lim_df_based_on_support(df,min_support):

    for method in ["sobol","ehvi","parego"]:
        mask = (df[f"{method}_count"] >= min_support).to_list()

        if not all(mask):
            max_idx = mask.index(False)
            df.iloc[max_idx:,col_idx(df,method)] = None



# In[6]:


def save_new_df(df,prefix,min_support,up_to_complexity):

    if up_to_complexity:
        new_name = prefix.replace("_seed",f"_avgs_adjusted_up_to_complexity_support{min_support}.csv")
    else:
        new_name = prefix.replace("_seed",f"_avgs_adjusted_support{min_support}.csv")
    df.drop([f"{method}_count" for method in ["sobol","ehvi","parego"]],axis=1).to_csv(new_name)


# In[7]:


def lim_ax_methods_dfs_with_counts_and_save(min_support,up_to_complexity):

    ax_results_folder = "/home/lucas/Documents/perceptronac/complexity/scripts/ax_experiments/ax_results_energy_params_bits/"

    adjusted_data_folder = "/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/"
    
    for x_axis in ["micro_joules_per_pixel","model_bits","params"]:
        
        print(x_axis)

        prefix = f"{x_axis}_data_bits_over_data_samples_ax_methods_seed"
        
        stats_dict = ax_methods_trained_networks(ax_results_folder,prefix)

        df = ax_methods_df_with_counts(adjusted_data_folder,prefix,stats_dict,up_to_complexity)

        lim_df_based_on_support(df,min_support)
        
        save_new_df(df,prefix,min_support,up_to_complexity)



# In[8]:


def lim_ax_methods_dfs_with_counts_and_save_rdc(min_support,up_to_complexity):

    adjusted_data_folder = "/home/lucas/Documents/perceptronac/complexity/scripts/glch_experiments/"
    
    for x_axis in ["params","flops"]:
        
        print(x_axis)
        
        ax_results_folder = f"/home/lucas/Documents/perceptronac/complexity/scripts/ax_experiments/ax_results_rdc_{x_axis}"

        prefix = f"bpp_loss_mse_loss_{x_axis}_ax_methods_seed"
        
        stats_dict = ax_methods_trained_networks(ax_results_folder,prefix)

        df = ax_methods_df_with_counts(adjusted_data_folder,prefix,stats_dict,up_to_complexity)

        lim_df_based_on_support(df,min_support)
        
        save_new_df(df,prefix,min_support,up_to_complexity)







