import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x=np.array(x)
        W1=np.array(W1)
        b1=np.array(b1)
        W2=np.array(W2)
        b2=np.array(b2)
        z1=np.dot(W1,x) + b1
        a1=np.maximum(0,z1)
        relu_mask = (z1>0)
        y_true=np.array(y_true)

        y_hat=np.dot(W2,a1) + b2
        loss=np.mean((y_hat-y_true)**2)
        n=y_true.shape[0]

        dl_dyhat=2*(y_hat-y_true)/n

        dl_db2 = dl_dyhat
        dl_dW2 = np.outer(dl_dyhat,a1)

        dl_da1 = np.dot(np.transpose(W2),dl_dyhat)
        dl_dz1 = dl_da1*relu_mask

        dl_db1 = dl_dz1
        dl_dW1 = np.outer(dl_dz1 , x)
        return {"loss": round(float(loss),4) , "dW1" : np.round(dl_dW1,3).tolist() , "db1" : np.round(dl_db1 , 3).tolist() , "dW2" : np.round(dl_dW2 , 3).tolist() , "db2" : np.round(dl_db2 , 3).tolist()}
