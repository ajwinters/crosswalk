"""hyun woo kim, the pennsylvania state university, 2018-2019"""

import pandas
import pandas as pd
import numpy
import numpy as np
import seaborn as sns

  
def prep(df: pandas.DataFrame, *var: pandas.Series):
    
    """
    Prepare the ground-truth data or the predicted data for evaluations.
    
    Drop self-duplicates (such as comparison between 12345 and 12345) and
    transform the data to a upper triangle matrix.
    
    Parameters
    ----------
    var : pandas.Series
        The ground-truth predicted data frame
    var : pandas.Series
        The Pandas series of the common variable with which you want to replace
        (optional).

    Returns
    -------
    df : pandas.DataFrame
        The refined gold standard data.
    """

    #
    if var and len(var) > 0:
        
        df = df.reset_index()
        col0, col1 = df.columns[0], df.columns[1]
        df = df.merge(var[0], left_on = col0, right_index = True)
        df = df.merge(var[0], left_on = col1, right_index = True)
        df = df[[var[0].name + '_x', var[0].name + '_y']]

    #tidy up
    df = df.sort_values([df.columns[0], df.columns[1]])
    df = df[df[df.columns[0]].notna() & df[df.columns[1]].notna()]
    df[df.columns[0]] = pandas.to_numeric(df[df.columns[0]], errors='coerce')
    df[df.columns[1]] = pandas.to_numeric(df[df.columns[1]], errors='coerce')
    df = df.dropna().astype(int)
    df = df.dropna().astype(int)
    
    #drop self-duplicates
    df = df[df[df.columns[0]] != df[df.columns[1]]]
    
    #upper triangle matrix
    #df1 = df[df[df.columns[0]] < df[df.columns[1]]]
    #df2 = df[df[df.columns[0]] > df[df.columns[1]]]
    #df2 = df2.rename(columns = {df.columns[0]: df.columns[1],
    #                            df.columns[1]: df.columns[0]})
    #df = df1.append(df2, sort=True)
    df = df.drop_duplicates()
    
    #multiindex
    df.set_index([df.columns[0], df.columns[1]], inplace = True)
    df.sort_index(inplace = True)

    return df



def confusion_matrix(true_data, pred_data):

    """
    Place the gold standard data and the predicted data side-by-side and create
    a confusion matrix.
    
    It compares features of indexed observations and returns a Boolean table of
    them. First name is compared not only with first name, but also last name
    (and so is the case for last name) in order to handle transposed name input
    error. Uses recordlinkage as an external library. ``n_jobs'' is set to 6. 
    
    Parameters
    ----------
    true_data : pandas.DataFrame
        You know what it is.
    pred_data : pandas.DataFrame
        You know what it is too.

    Returns
    -------
    bmatrix : pandas.DataFrame
        Boolean matrix of feature comparison.
    fp : pandas.MultiIndex
        False Positive cases for clerical reviews
    fn : pandas.MultiIndex
        False Negative cases for clerical reviews
    """  
     
    
    #normal pandas.MultiIndex
    dfx = true_data.sort_index()
    dfy = pred_data.sort_index()
    
    #transposed pandas.MultiIndex (bypass a glitch in recordlinkage)
    dfxt = dfx.reset_index()
    dfxt = dfxt.rename(columns = {dfx.index.names[0]: dfx.index.names[1],
                                  dfx.index.names[1]: dfx.index.names[0]})
    dfxt.set_index([dfxt.columns[1], dfxt.columns[0]], inplace = True)
    dfyt = dfy.reset_index()
    dfyt = dfyt.rename(columns = {dfx.index.names[0]: dfx.index.names[1],
                                  dfx.index.names[1]: dfx.index.names[0]})
    dfyt.set_index([dfyt.columns[1], dfyt.columns[0]], inplace = True)
    
    #true postive or good match
    dfxy1 = dfx.index & dfy.index
    dfxy2 = dfx.index & dfyt.index
    tp = dfxy1.append(dfxy2).drop_duplicates()
    #[f for f in tp]
    
    #false positive or erroneous match
    sample = dfy.loc[dfx.index.levels[0]]
    fp = sample.index.difference(dfx.index).difference(dfxt.index)
    #[f for f in fp]
    
    #false negative or erroneous non-match
    dfY = dfy.index | dfyt.index
    fn = dfx.index.difference(dfY)
    #[f for f in fn]

    #true negative or good non-match: len(Data A), len(Data B)
    tn = int(len(dfx) * dim) - len(fp) - len(fn) - len(tp)
    
    #confusion matrix
    cmat = numpy.matrix([[tn,      len(fp)],
                         [len(fn), len(tp)]])
    
    return cmat, fp, fn


def rule_of_thumb(df, features):

    """
    Internal use only
    """
    
    #rule-of-thumb cutpoint
    rot_cutp = 0
    for feature in features:
        rot_cutp = rot_cutp + df[df[feature]>0][feature].mean()
    
    return rot_cutp


def grids(df, features, bound = 10):

    """
    Return a range of grids for cutoff values.
    
    This is for a hyperparameter optimization. See Geron (2018).
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dyad data that contain weight factors.
    features : list
        The list of representative features to calculate the rule of thumb
    bound : int
        How far around the rule-of-thumb cutpoint you want to conduct grid
        searches. Default is 10.

    Returns
    -------
    grid : numpy.ndarray
        The grid of cutoff values to be searched
    """
    
    rot_cutp = rule_of_thumb(df, features)
    
    #define a linear space around the rule-of-thumb cutpoint
    grids = numpy.linspace(rot_cutp - bound, rot_cutp + bound)
    
    return grids


def precision_recall(cmat):
    
    #accuracy or (TP + TN) / (TP + TN + FP + FN)   
    accuracy = (cmat[1,1] + cmat[0,0]) / cmat.sum()
    
    #the precision is given by TP/(TP+FP)
    prec = cmat[1,1] / cmat[:, 1].sum()
    
    #the recall is given by TP/(TP+FN)
    rec = cmat[1,1] / cmat[1:].sum()
    
    #the f1 score is given by 2 * (precision*recall) / (precision+recall)
    f1 = 2*prec*rec / (prec + rec)
    
    return accuracy, prec, rec, f1
    


def cutoff(cmat):

    """
    Return the best cutoff value.
    
    This is for a hyperparameter optimization. See Geron (2018).
    
    Parameters
    ----------
    cmat : numpy.matrix
        Confusion matrix.

    Returns
    -------
    best_cutoff : float
        The cutoff value that achieves the highest F1 score.
    best_score : float
        The F1-score that was achieved by the best cutoff.
    """

    scores = []
    for i in cmat.keys():
        accuracy, prec, rec, f1 = precision_recall(cmat[i][0])
        scores.append(f1)
    pos = numpy.argmax(scores)
    best_cutoff = list(cmat.keys())[pos]
    best_score = scores[pos]
    
    return best_cutoff, best_score
        

def clerical_review(df, m):

    """
    Return false positive cases and false negative cases for clerical reviews.
    
    Clerical Review!
    
    Parameters
    ----------
    df : pandas.DataFrame
        The data set
    m : tuple
        The result of ``crosswalk.cpu.evaluation.confusion_matrix''

    Returns
    -------
    fp : pandas.DataFrame
        False positive cases
    fn : pandas.DataFrame
        False negative cases
    """
    
    
    #false positive
    fp_list = []
    colname = m[1].names[0][:-2]
    for i in m[1]:
        fp_list.append(df[df[colname]==i[0]].reset_index())
        fp_list.append(df[df[colname]==i[1]].reset_index())
        fp_list.append(pd.Series(dtype='float64'))
    fp = pd.concat(fp_list, ignore_index=True)

    #false negative
    fn_list = []
    colname = m[2].names[0][:-2]
    for i in m[2]:
        fn_list.append(df[df[colname]==i[0]].reset_index())
        fn_list.append(df[df[colname]==i[1]].reset_index())
        fn_list.append(pandas.Series(dtype='float64'))
        fn = pd.concat(fn_list, ignore_index=True)
        fn_list.append(pandas.Series())
    fn = pandas.concat(fn_list, ignore_index=True)
    
    return fp, fn
    

def pr_tradeoff(cmat, filename):

    #geron (2018: 92)
    rec =  [precision_recall(cmat[i])[2] for i in cmat.keys()]
    prec = [precision_recall(cmat[i])[1] for i in cmat.keys()]
    
    return rec, prec
    #return [i for i in zip(rec, prec)]
    
    seaborn.scatterplot(a, b)
    seaborn.set(rc={'figure.figsize': (15, 15)})
    seaborn.lineplot(data=pandas.DataFrame(a,b), linewidth=0.5)
    seaborn.scatterplot(data=pandas.DataFrame(a,b))
    

def roc_curve(cmat, filename):

    """
    roc curve is meaningless if deduplication because false FP rate is too
    small.
    """
    #false positive rate is fallout
    fpr = [cmat[i][0,1] / cmat[i][0].sum() for i in cmat.keys()]
    
    #true positive rate or precision
    tpr = [precision_recall(cmat[i])[1] for i in cmat.keys()]
    
    return fpr, tpr

    #return [i for i in zip(fpr, tpr)]
    
 


def old_index(df, idx):

    """
    Merge an index to a data set.
    
    Just in case the gold standard data and the predicted data do not have a
    common index, a common index should be merged to both of them.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The predicted data or the gold standard data
    idx : pandas.DataFrame
        The first column should contain an index ``df'' uses. The second column
        should contain the index both of data sets use in common.

    Returns
    -------
    df : pandas.DataFrame
        The predicted data or the gold standard data with the common index.
    """
    
    #merge index with recordpersonid
    df = df.merge(idx, left_on = df.columns[0], right_on = 'index')
    df = df.merge(idx, left_on = df.columns[1], right_on = 'index')
    df = df[['recordpersonid_x', 'recordpersonid_y']]
    
    #drop duplicate dyads
    df = df[df['recordpersonid_x'] != df['recordpersonid_y']]

    #upper triangle matrix
    df1 = df[df['recordpersonid_x'] <= df['recordpersonid_y']]
    df2 = df[df['recordpersonid_x'] > df['recordpersonid_y']]
    df2 = df2.rename(columns = {'recordpersonid_x': 'recordpersonid_y',
                                'recordpersonid_y': 'recordpersonid_x'})
    df = df1.append(df2, sort=True)
    
    #multiindex
    df.set_index(['recordpersonid_x', 'recordpersonid_y'], inplace = True)
    df.sort_index(inplace = True)

    return df

  

def old_confusion_matrix(true_data, pred_data, review = False):
  
    """
    Place the gold standard data and the predicted data side-by-side and create
    a confusion matrix.
    
    It compares features of indexed observations and returns a Boolean table of
    them. First name is compared not only with first name, but also last name
    (and so is the case for last name) in order to handle transposed name input
    error. Uses recordlinkage as an external library. ``n_jobs'' is set to 6. 
    
    Parameters
    ----------
    true_data : pandas.DataFrame
        You know what it is.
    pred_data : pandas.DataFrame
        You know what it is too.
    review : Boolean
        True if you want to false positive and false negative cases for
        clerical reviews. Default is False.

    Returns
    -------
    bmatrix : pandas.DataFrame
        Boolean matrix of feature comparison.
    """  
     
    
    #
    dfx = true_data.sort_index()[:5]
    fake = pandas.DataFrame({'recnumbr_x': ["40170453"],
                             'recnumbr_y': ["test"]})
    fake.set_index(['recnumbr_x', 'recnumbr_y'], inplace = True)
    dfx = dfx.append(fake)
    
    #normal pandas.MultiIndex
    dfy = pred_data.sort_index()
    dfxy1 = dfx.index & dfy.index

    #transposed pandas.MultiIndex
    #dfy.index.names=list(['recnumbr_y', 'recnumbr_x'])
    dfy = dfy.reset_index()
    dfy.rename(columns = {'recnumbr_x': 'recnumbr_y',
                          'recnumbr_y': 'recnumbr_x'}, inplace = True)
    dfy.set_index(['recnumbr_x', 'recnumbr_y'], inplace = True)
    #index.names=list(['recnumbr_y', 'recnumbr_x'])
    dfxy2 = dfx.index & dfy.index
    
    #true positive
    [i for i in dfx.index]
    dfxy = dfxy1.append(dfxy2)
    tp = [f for f in dfxy.drop_duplicates()]
    len(tp)
    
    #a subset of predicted data corresponding to the gold standard dataset
    #subset = pred_data.loc[sample]
    #dfxy = true_data.merge(subset, left_index = True, right_index = True,
    #                          how = 'outer')

    #true postive or good match
    tp = dfxy[(dfxy['true']==1) & (dfxy['pred']==1)]
    
    #false positive or erroneous match
    fp = dfxy[(dfxy['true'].isna()) & (dfxy['pred']==1)]
    
    #false negative or erroneous non-match
    fn = dfxy[(dfxy['true']==1) & (dfxy['pred'].isna())]

    #true negative or good non-match
    U = [len(dfx), dim]         #len(Data A), len(Data B)
    tn = dfxy[(dfxy['true'].isna()) & (dfxy['pred'].isna())]
    tn_alt = int(U[0] * U[1]) - len(fp) - len(fn) - len(tp)
    
    #confusion matrix
    cmat = numpy.matrix([[tn_alt,  len(fp)],
                         [len(fn), len(tp)]])
    
    #for clerical reviews
    if review == True:
        return cmat, fp, fn
    
    if review == False:
        return cmat

