
import torch
import numpy as np
import numbers
import pandas as pd
import os
import matplotlib.pyplot as plt


def plot_single_curve(ax,rates,xvalues,linestyle,color,label,marker):

    handle = None
    if (len(set(rates)) == 1) and (len(rates) != 1):
        marker = ""
    valid_indices = [i for i,v in enumerate(rates) if v != -1]
    if valid_indices:            
        handle,= ax.plot(
            [xvalues[i] for i in valid_indices], 
            [rates[i] for i in valid_indices], 
            linestyle=linestyle,color=color, 
            label=label,marker=marker
        )
    return handle


def plot_comparison(xvalues,data,xlabel,xscale="linear",linestyles=None,colors=None,markers=None):

    if linestyles is None:
        linestyles = ["dashdot","dotted","solid","dashed"]
    if colors is None:
        colors = ['red','green','blue','orange']
    if markers is None:
        markers = ['s','^','o','v']

    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(4.8,4.8))    

    handles = []
    for k,linestyle,color,marker in zip(sorted(data.keys()),linestyles,colors,markers):
        handle = plot_single_curve(ax,data[k],xvalues,linestyle,color,k,marker)
        if handle:
            handles.append(handle)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('bits/sample')
    ax.set_xscale(xscale)
    ax.legend(handles=handles,loc="upper right")

    fig.tight_layout()
    plt.show()    
    return fig


def save_values(csv_name,xvalues,data,xlabel):
    csv_name = os.path.splitext(csv_name)[0]
    values = pd.DataFrame(data)
    values.index = xvalues
    values.index.name = xlabel
    values.to_csv(f"{csv_name}.csv")


def save_configs(csv_name,configs):
    csv_name = os.path.splitext(csv_name)[0]
    df=pd.DataFrame([{k:(v if isinstance(v,numbers.Number) else str(v)) for k,v in configs.items()}])
    df = df.T
    df.columns = ["value"]
    df.index.name = "key"
    df.to_csv(f"{csv_name}.csv")


def save_fig(fig_name,fig):
    fig_name = os.path.splitext(fig_name)[0]
    fig.savefig(f"{fig_name}.png", dpi=300)
    
    
def save_model(file_name,model):
    file_name = os.path.splitext(file_name)[0]
    torch.save(model.eval().state_dict(), f"{file_name}.pt")
    

def get_prefix(configs, id_key = 'id' ):
    return f"{configs['save_dir'].rstrip('/')}/exp_{configs[id_key]}/exp_{configs[id_key]}"


def load_model(configs,N):
    ModelClass=configs["ModelClass"]
    model = ModelClass(N)
    if configs.get("parent_id"):
        if ('train' not in configs["phases"]) and (configs["reduction"] == 'min'):
            file_name = f"{get_prefix(configs,'parent_id')}_{N:03d}_min_valid_loss_model.pt"
        else:
            file_name = f"{get_prefix(configs,'parent_id')}_{N:03d}_model.pt"
        print(f"loading file {file_name}")
        model.load_state_dict(torch.load(file_name))
    return model


def save_N_min_valid_loss_model(valid_loss,configs,N,mlp_model):
    if len(valid_loss) == 0:
        pass
    elif (min(valid_loss) == valid_loss[-1]) and ('train' in configs["phases"]) and (N>0):
        save_model(f"{get_prefix(configs)}_{N:03d}_min_valid_loss_model",mlp_model)


def save_N_model(configs,N,mlp_model):
    if ('train' in configs["phases"]) and (N>0):
        save_model(f"{get_prefix(configs)}_{N:03d}_model",mlp_model)
    

def save_N_data(configs,N,N_data):
    
    xvalues = np.arange(configs["epochs"])
    xlabel = "epoch"

    for phase in configs["phases"]:
        
        fig = plot_comparison(xvalues,N_data[phase],xlabel)
        save_fig(f"{get_prefix(configs)}_{N:03d}_{phase}_graph",fig)
        save_values(f"{get_prefix(configs)}_{N:03d}_{phase}_values",xvalues,N_data[phase],xlabel)
 

def save_final_data(configs,data):
    
    xvalues = configs["N_vec"]
    xlabel = "context size"
    xscale = configs["xscale"]
    
    save_configs(f"{get_prefix(configs)}_conf",configs)
    
    for phase in configs["phases"]:
        
        fig=plot_comparison(xvalues,data[phase],xlabel,xscale=xscale)
        save_fig(f"{get_prefix(configs)}_{phase}_graph",fig)
        save_values(f"{get_prefix(configs)}_{phase}_values",xvalues,data[phase],xlabel)
