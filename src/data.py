import numpy as np

def standardize(x):
    """Outputs the matrix x after normalization."""
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

def get_proportion(y):
    """Get proportion of different labels"""
    proportion_a=float(np.count_nonzero(y == 1))/float(y.shape[0])
    proportion_b=1-proportion_a
    print(f'Porprtion label a = {proportion_a} \nProportion label b = {proportion_b}')
    return proportion_a , proportion_b

def empty_features(tX):
    """Get proportion of empty cell in each features"""
    empty_cell=np.count_nonzero(tX==-999, axis=0)
    return empty_cell/float(tX.shape[0])

def split_in_labels(y,tX,label_b=-1):
    """Split tX whith respect to labels"""
    ind_label_a=np.where(y==1)
    ind_label_b=np.where(y==label_b)
    tX_label_a=tX[ind_label_a,:]
    tX_label_b=tX[ind_label_b,:]
    return np.squeeze(tX_label_a) , np.squeeze(tX_label_b)

def anova_test(tX_a,tX_b):
    """Anova test on features to see which one are relevant"""
    mean_labels=[np.mean(tX_a,0), np.mean(tX_b,0)]
    #between the sum of square (variance between the mean of each groups)
    SSB=np.var(mean_labels,0)
    #within the sum of square (variance beween)
    SSW=np.var(tX_a,0)+np.var(tX_b,0)
    F=SSB/SSW
    print(np.argsort(F))
    #print(F)
    return F

def remove_features(tX,indice):
    """remove feature at list of indice on tX"""
    return np.delete(tX,indice,1)

def replace_empty_with_mean(tX):
    """replace -999 with mean of true values"""
    for i in range(0,tX.shape[1]):
        ind_full_cell=np.where(tX[:,i]!=-999)
        mean_feature=np.mean(tX[ind_full_cell,i])
        ind_empty_cell=np.where(tX[:,i]==-999)
        tX[ind_empty_cell,i]=mean_feature
    return tX



        

        
        