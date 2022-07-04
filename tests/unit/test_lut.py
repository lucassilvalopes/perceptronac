
# %% [markdown]
# Useful links:
# 
# - https://github.com/mohammadimtiazz/standard-test-images-for-Image-Processing
# - https://stackoverflow.com/questions/68795553/find-index-of-a-row-in-numpy-array

# %%
import numpy as np
from perceptronac.utils import causal_context_many_imgs
from perceptronac.context_training import context_training_nonbinary
from perceptronac.context_coding import context_coding_nonbinary

# %%
def test_context_training_nonbinary():

    test_X = np.vstack(list(map(lambda x:x.reshape(-1),np.meshgrid(range(3),range(3),range(3))))).T
    test_X = test_X[np.lexsort((test_X[:, 2], test_X[:, 1], test_X[:, 0]))]
    test_y = test_X[::-1,:]

    test_table,test_contexts=context_training_nonbinary(test_X,test_y)

    assert np.allclose(test_contexts,test_X)

    assert np.all( test_table[:,3:,:] == 0 )

    assert np.allclose(np.sum( test_table[:,:3,:] * np.array([[[0],[1],[2]]]) , axis=1) , test_y)

# %%
if __name__ == "__main__":
    # !wget https://raw.githubusercontent.com/mohammadimtiazz/standard-test-images-for-Image-Processing/master/standard_test_images/baboon.png
    # !wget https://raw.githubusercontent.com/mohammadimtiazz/standard-test-images-for-Image-Processing/master/standard_test_images/fruits.png
# %%
    yt,Xt = causal_context_many_imgs(pths=["baboon.png"],N=1,color_mode="rgb")
    yc,Xc = causal_context_many_imgs(pths=["fruits.png"],N=1,color_mode="rgb")
    table, contexts = context_training_nonbinary(Xt,yt)
    preds = context_coding_nonbinary(Xc,table,contexts)