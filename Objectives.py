import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gower
from scipy.sparse.csgraph import minimum_spanning_tree


class Objectives():

  def branch(X,y,j):
    return X[np.where(y==j)[0],:]

  def numerator(X,y,j):
    tmp = Objectives.branch(X,y,j)
    aux = tmp.shape[0] * (np.mean(tmp,axis=0) - np.mean(X, axis=0))**2
    return aux

  def denominator(X,y,j):
    tmp = Objectives.branch(X,y,j)
    aux = np.sum((tmp-np.mean(tmp,axis=0))**2,axis=0)
    return aux

  def F1(X, y):
    num = []
    den = []
    for j in range(np.unique(y).shape[0]):
      num.append(Objectives.numerator(X,y,j))
      den.append(Objectives.denominator(X,y,j))
    
    aux = np.nansum(np.vstack(num),axis=0)/np.nansum(np.vstack(den),axis=0)
    F1 = 1/(aux+1)
    return np.nanmean(F1)

  def N1(X, y):
    # 0-1 scaler
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    X_ = scaler.transform(X)

    # compute the distance matrix and the minimum spanning tree.
    dist_m = np.triu(gower.gower_matrix(X_), k=1)
    mst = minimum_spanning_tree(dist_m)
    node_i, node_j = np.where(mst.toarray() > 0)

    # which edges have nodes with different class
    which_have_diff_cls = y[node_i] != y[node_j]

    # number of different vertices connected
    aux = np.unique(np.concatenate([node_i[which_have_diff_cls],node_j[which_have_diff_cls]])).shape[0]
    N1 = aux/X.shape[0]
    return N1