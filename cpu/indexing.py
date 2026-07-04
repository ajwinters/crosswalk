"""hyun woo kim, the pennsylvania state university, 2018-2019"""

import pandas, recordlinkage, jellyfish
        
def dedupe(indexer, transposed, df):
     
    """
    Conduct indexing for deduplication.
    
    It allows to pool multiple indices into a grand index. Transposed names
    (e.g. JOHN SMITH and SMITH JOHN) can be considered.
    
    Parameters
    ----------
    index : list
        A list of lists containing features you want to use.
    transposed : list
        A list of lists containing features you want to consider transposition.
    df : pandas.DataFrame
        Data set you want to index.

    Returns
    -------
    cpairs : pandas.multindex
        Index data for deduplication.
    """

    #empty start
    cpairs = pandas.MultiIndex(levels=[[], []], codes=[[], []])

    #normal indexers
    for i in indexer:
    
        Indexer = recordlinkage.Index()
        Indexer.block(i)
        cpairs_tba = Indexer.index(df)
        cpairs = cpairs.append(cpairs_tba)
    
    #transposed indexers
    if len(transposed) > 0:
        for j in transposed:
            
            Indexer = recordlinkage.Index()
            Indexer.block(left_on = [j[0], j[1]], right_on = [j[1], j[0]])
            cpairs_tba = Indexer.index(df)
            cpairs = cpairs.append(cpairs_tba)
    
    return cpairs.drop_duplicates()


def match(indexer, transposed, *dframes):
   
    """
    Conduct indexing for matching.
    
    It allows to pool multiple indices into a grand index. Transposed names
    (e.g. JOHN SMITH and SMITH JOHN) can be considered.
    
    Parameters
    ----------
    index : list
        A list of lists containing features you want to use.
    transposed : list
        A list of lists containing features you want to consider transposition.
    *dframes : pandas.DataFrame
        Two data sets you want to index.

    Returns
    -------
    cpairs : pandas.multindex
        Index data for matching.
    """

    dfA, dfB = dframes[0], dframes[1]
    
    #empty start
    cpairs = pandas.MultiIndex(levels=[[], []], codes=[[], []])

    #normal indexers
    for i in indexer:
    
        Indexer = recordlinkage.Index()
        Indexer.block(left_on = i, right_on = i)
        cpairs_tba = Indexer.index(dfA, dfB)
        cpairs = cpairs.append(cpairs_tba)
    
    #transposed indexers
    for j in transposed:
        
        Indexer = recordlinkage.Index()
        Indexer.block(left_on = [j[0], j[1]], right_on = [j[1], j[0]])
        cpairs_tba = Indexer.index(dfA, dfB)
        cpairs = cpairs.append(cpairs_tba)
    
    return cpairs.drop_duplicates()
