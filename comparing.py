"""hyun woo kim, the pennsylvania state university, 2018-2019"""
    
    
    
    
import recordlinkage, numpy



    
def dedupe(candidate_pairs, features, df):

    """
    Conduct comparison for deduplication.
    
    It compares features of indexed observations and returns the Boolean table
    of them. First name is compared not only with first name, but also last
    name (and so is the case for last name) in order to handle transposed name
    input error. Uses recordlinkage as an external library. n_jobs is set to 6. 
    
    Parameters
    ----------
    candidate_pairs : pandas.multiindex
        Index file created from crosswalk.indexing.dedupe
    features : a list of two lists
        The first list should contain features and the second one should
        contain the slice position of interaction terms in df.
    df : pandas.DataFrame
        Data frame you want to deduplicate.

    Returns
    -------
    bmatrix : pandas.DataFrame
        Boolean matrix of feature comparison.
    """

    #initiation
    comparer = recordlinkage.Compare(n_jobs = 6)

    #jaro-winkler is faster than levenshtein or damerau-levenshtein
    comparer.string('firstname', 'firstname',                 #df[0]
                    method = 'jarowinkler', threshold = 0.9)
    comparer.string('lastname', 'lastname',                   #df[1]
                    method = 'jarowinkler', threshold = 0.9)
    comparer.string('firstname', 'lastname',                  #df[2]
                    method = 'jarowinkler', threshold = 0.9)
    comparer.string('lastname', 'firstname',                  #df[3]
                    method = 'jarowinkler', threshold = 0.9)
    comparer.exact('suffix', 'suffix')                        #df[4]
    
    #features in addition to first name, last name and suffix
    for feature in features[0]:
        comparer.exact(feature, feature)
    
    #compute
    bmatrix = comparer.compute(candidate_pairs, df)
    bmatrix = bmatrix.astype(int)
    
    #handle the cross-field name comparison
    bmatrix[0] = numpy.where((bmatrix[2] == 1) & (bmatrix[3] == 1),
                             1, bmatrix[0])
    bmatrix[1] = numpy.where((bmatrix[2] == 1) & (bmatrix[3] == 1),
                             1, bmatrix[1])
    for i in range(2, len(bmatrix.columns) - 2):
        bmatrix[i] = bmatrix[(i + 2)]
    bmatrix.drop(columns = [bmatrix.columns[(-1)], bmatrix.columns[(-2)]],
                 inplace = True)
    
    #interaction terms
    if len(features[1][0]) > 0:         #if any
        for it in features[1]:
            bmatrix[len(bmatrix.columns)] = bmatrix[it].min(axis = 1)

    return bmatrix



def match(candidate_pairs, features, *dframes):

    """
    Conduct comparison for matching.
    
    It compares features of indexed observations and returns the Boolean table
    of them. First name is compared not only with first name, but also last
    name (and so is the case for last name) in order to handle transposed name
    input error. Uses recordlinkage as an external library. n_jobs is set to 6. 
    
    Parameters
    ----------
    candidate_pairs : pandas.multiindex
        Index file created from crosswalk.indexing.dedupe
    features : a list of two lists
        The first list should contain features and the second one should
        contain the slice position of interaction terms in df.
    df : pandas.DataFrame
        Data frame you want to deduplicate.

    Returns
    -------
    bmatrix : pandas.DataFrame
        Boolean matrix of feature comparison.
    """

    dfA, dfB = dframes[0], dframes[1]

    #initiation
    comparer = recordlinkage.Compare(n_jobs = 6)

    #jaro-winkler is faster than levenshtein or damerau-levenshtein
    comparer.string('firstname', 'firstname',
                    method = 'jarowinkler', threshold = 0.9)
    comparer.string('lastname', 'lastname',
                    method = 'jarowinkler', threshold = 0.9)
    comparer.string('firstname', 'lastname',
                    method = 'jarowinkler', threshold = 0.9)
    comparer.string('lastname', 'firstname',
                    method = 'jarowinkler', threshold = 0.9)
    comparer.exact('suffix', 'suffix')
    
    #features in addition to first name, last name and suffix
    for feature in features[0]:
        comparer.exact(feature, feature)

    #compute
    bmatrix = comparer.compute(candidate_pairs, dfA, dfB)
    bmatrix = bmatrix.astype(int)

    #handle the cross-field name comparison
    bmatrix[0] = numpy.where((bmatrix[2] == 1) & (bmatrix[3] == 1),
                             1, bmatrix[0])
    bmatrix[1] = numpy.where((bmatrix[2] == 1) & (bmatrix[3] == 1),
                             1, bmatrix[1])
    for i in range(2, len(bmatrix.columns) - 2):
        bmatrix[i] = bmatrix[(i + 2)]
    bmatrix.drop(columns = [bmatrix.columns[(-1)], bmatrix.columns[(-2)]],
                 inplace = True)
        
    #interaction terms
    if len(features[1][0]) > 0:         #if any
        for it in features[1]:
            bmatrix[len(bmatrix.columns)] = bmatrix[it].min(axis = 1)

    return bmatrix







