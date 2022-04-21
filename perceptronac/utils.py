import torch
import numpy as np
from PIL import Image
from netpbmfile import imwrite
import numbers
import pandas as pd
import os
import subprocess as sb
import matplotlib.pyplot as plt
from perceptronac.coding2d import causal_context
import perceptronac.coding3d as c3d


def load_model(configs,N):
    ModelClass=configs["ModelClass"]
    model = ModelClass(N)
    if configs.get("parent_id"):
        if ('train' not in configs["phases"]) and (configs["reduction"] == 'min'):
            file_name = \
                f"results/exp_{configs['parent_id']}/exp_{configs['parent_id']}_{N:03d}_min_valid_loss_model.pt"
        else:
            file_name = f"results/exp_{configs['parent_id']}/exp_{configs['parent_id']}_{N:03d}_model.pt"
        print(f"loading file {file_name}")
        model.load_state_dict(torch.load(file_name))
    return model


def add_border(img,N):
    ns = int(np.ceil(np.sqrt(N)))
    nr,nc = img.shape[:2]
    new_img = 255*np.ones((nr+ns,nc+2*ns))
    new_img[ns:nr+ns,ns:nc+2*ns-ns] = img.copy()
    return new_img


def causal_context_many_imgs(pths,N):
    y = []
    X = []
    for pth in pths:
        img = add_border(read_im2bw(pth,0.4),N)
        partial_y,partial_X = causal_context(img, N)
        y.append(partial_y)
        X.append(partial_X)
    y = np.vstack(y)
    X = np.vstack(X)
    return y,X


def causal_context_many_pcs(pths,N,percentage_of_uncles):
    y = []
    X = []

    M = int(percentage_of_uncles * N)
    print(f"using {N-M} siblings and {M} uncles.")
    for pth in pths:
        pc = c3d.read_PC(pth)[1]
        _,partial_X,partial_y,_,_ = c3d.pc_causal_context(pc, N-M, M)
        y.append(np.expand_dims(partial_y.astype(int),1) )
        X.append(partial_X.astype(int))
    y = np.vstack(y)
    X = np.vstack(X)
    return y,X


def plot_single_fig(
    ax,data,xvalues,xlabel,
    xscale="linear",mlp_marker='o',static_marker='v',cabac_marker='^',
    jbig1_marker="s"
):
    
    rates_mlp = data.get("mlp")
    rates_static = data.get("static")
    rates_cabac = data.get("cabac")
    rates_jbig1 = data.get("jbig1")
            
    handles = []
    if rates_mlp:
        if (len(rates_mlp) == 1) and (len(xvalues) != 1):
            rates_mlp = rates_mlp[0] * np.ones(len(xvalues))
            mlp_marker = ""
        mlp_handle,= ax.plot(
            xvalues, rates_mlp, 
            linestyle="solid",color='blue',
            label='mlp',marker=mlp_marker
        )
        handles.append(mlp_handle)
    if rates_static:
        if (len(rates_static) == 1) and (len(xvalues) != 1):
            rates_static = rates_static[0] * np.ones(len(xvalues))
            static_marker = ""
        static_handle,= ax.plot(
            xvalues, rates_static, 
            linestyle="dashed",color='orange',
            label='static',marker=static_marker
        )
        handles.append(static_handle)        
    if rates_cabac:
        if (len(rates_cabac) == 1) and (len(xvalues) != 1):
            rates_cabac = rates_cabac[0] * np.ones(len(xvalues))
            cabac_marker = ""
        valid_indices = [i for i,v in enumerate(rates_cabac) if v != -1]
        if valid_indices:            
            cabac_handle,= ax.plot(
                [xvalues[i] for i in valid_indices], 
                [rates_cabac[i] for i in valid_indices], 
                linestyle="dotted",color='green', 
                label='cabac',marker=cabac_marker
            )
            handles.append(cabac_handle)
    if rates_jbig1:
        if (len(rates_jbig1) == 1) and (len(xvalues) != 1):
            rates_jbig1 = rates_jbig1[0] * np.ones(len(xvalues))
            jbig1_marker = ""
        valid_indices = [i for i,v in enumerate(rates_jbig1) if v != -1]
        if valid_indices:            
            jbig1_handle,= ax.plot(
                [xvalues[i] for i in valid_indices], 
                [rates_jbig1[i] for i in valid_indices], 
                linestyle="dashdot",color='red', 
                label='jbig1',marker=jbig1_marker
            )
            handles.append(jbig1_handle)
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel('bits/sample')
    ax.set_xscale(xscale)
    ax.legend(handles=handles,loc="upper right")


def plot_comparison(
    xvalues,data,xlabel,
    xscale="linear",mlp_marker='o',static_marker='v',cabac_marker='^',jbig1_marker="s"
):

    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(4.8,4.8))    
    plot_single_fig(
        ax,data,xvalues,xlabel,xscale,
        mlp_marker, static_marker, cabac_marker, jbig1_marker
    )
    fig.tight_layout()
    plt.show()    
    return fig


def save_values(csv_name,xvalues,data,xlabel):
    csv_name = os.path.splitext(csv_name)[0]
    data = {k: (v[0] * np.ones(len(xvalues)) if len(v) == 1 else v) for k,v in data.items()} 
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
    

def save_N_min_valid_loss_model(valid_loss,configs,N,mlp_model):
    if len(valid_loss) == 0:
        pass
    elif (min(valid_loss) == valid_loss[-1]) and ('train' in configs["phases"]) and (N>0):
        save_model(f"results/exp_{configs['id']}/exp_{configs['id']}_{N:03d}_min_valid_loss_model",mlp_model)


def save_N_model(configs,N,mlp_model):
    if ('train' in configs["phases"]) and (N>0):
        save_model(f"results/exp_{configs['id']}/exp_{configs['id']}_{N:03d}_model",mlp_model)
    

def save_N_data(configs,N,N_data):
    
    common_name = f"results/exp_{configs['id']}/exp_{configs['id']}_{N:03d}"
    xvalues = np.arange(configs["epochs"])
    xlabel = "epoch"
    phases=configs["phases"]
        
    for phase in phases:
        
        fig = plot_comparison(xvalues,N_data[phase],xlabel)
        save_fig(f"{common_name}_{phase}_graph",fig)
        save_values(f"{common_name}_{phase}_values",xvalues,N_data[phase],xlabel)
 

def save_final_data(configs,data):
    
    common_name = f"results/exp_{configs['id']}/exp_{configs['id']}"
    xvalues = configs["N_vec"]
    xlabel = "context_size"
    phases=configs["phases"]
    xscale = configs["xscale"]
    
    save_configs(f"{common_name}_conf",configs)
    
    for phase in phases:
        
        filtered_data = {k:v for k,v in data[phase].items() if k !="static"}

        fig=plot_comparison(xvalues,filtered_data,xlabel,xscale=xscale)
        save_fig(f"{common_name}_{phase}_graph",fig)
        save_values(f"{common_name}_{phase}_values",xvalues,filtered_data,xlabel)


def save_pbm(file_name, binary_image):
    """
    
    Args:
        binary_image : numpy array representing the binary_image
    """
    
    data = binary_image.astype(np.uint8)
    imwrite(file_name, data,maxval=1)


def jbig1_rate(im_path):
    
#     src_path = "ieee_tip2017_klt1024_3.pbm"
#     dst_path = "ieee_tip2017_klt1024_3.jbg"

    src_path = "/tmp/tmp.pbm"
    dst_path = "/tmp/tmp.jbg"
    
    h,w = im2pbm(im_path,src_path,0.4)
    sb.run(["pbmtojbg","-q",src_path,dst_path])
    sz = os.path.getsize(dst_path) # bytes

    rate = 8 * sz / (w * h)
    
    os.remove(src_path)
    os.remove(dst_path)
    
    return rate


def read_im2bw(file_name,level):
    """
    Function with equivalent behaviour to matlab's im2bw
    
    Args:
        file_name : path to the image file
        level : threshold
    
    Returns:
        : numpy array representing the binary image
    """

    img = Image.open(file_name)
    
    thresh = level*255
    fn = lambda x : 255 if x > thresh else 0
    r = img.convert('L').point(fn, mode='1')
    return np.array(r).astype(int)


def im2pbm(im_path,pbm_path,th = 0.4):
    im = read_im2bw(im_path,th)
    save_pbm(pbm_path, im)
    return im.shape[:2]


if __name__ == "__main__":

    imgtraining = np.genfromtxt(
        "/home/lucas/Desktop/computer_vision/3DCNN/perceptron_ac/imgtraining.txt", 
        delimiter=",")
    imgcoding = np.genfromtxt(
        "/home/lucas/Desktop/computer_vision/3DCNN/perceptron_ac/imgcoding.txt", 
        delimiter=",")

    imgtraining_2 = read_im2bw(
        '/home/lucas/Desktop/computer_vision/3DCNN/perceptron_ac/images/ieee_tip2017_klt1024_3.png',
        0.4)
    imgcoding_2 = read_im2bw(
        '/home/lucas/Desktop/computer_vision/3DCNN/perceptron_ac/images/ieee_tip2017_klt1024_5.png',
        0.4)

    assert np.all(imgtraining == imgtraining_2)
    assert np.all(imgcoding == imgcoding_2)