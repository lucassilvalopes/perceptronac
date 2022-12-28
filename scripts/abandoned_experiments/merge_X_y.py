import numpy as np
import sys
from tqdm import tqdm

def merge_X_y(X_path,y_path,Xy_path):

    X = np.genfromtxt(X_path)

    y = np.genfromtxt(y_path)

    y = y.reshape(-1,1)

    Xy = np.concatenate([X,y],axis=1)

    np.savetxt(Xy_path, Xy, fmt="%d")


if __name__ == "__main__":

    X_path = "/home/lucas/Documents/data/eduardo/{}_frame000{}_contexts.txt"
    
    y_path = "/home/lucas/Documents/data/eduardo/{}_frame000{}_symbols.txt"
    
    Xy_path = "/home/lucas/Documents/data/eduardo/{}_frame000{}_contexts_symbols.txt"


    people = ["andrew","david","phil","ricardo","sarah"]

    frames = list(range(0,10))

    pbar = tqdm(total=len(people)*len(frames))
    for person in people:
        for frame in frames:
            merge_X_y(X_path.format(person,frame),y_path.format(person,frame),Xy_path.format(person,frame))
            pbar.update(1)
    pbar.close()