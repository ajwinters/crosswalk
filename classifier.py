"""hyun woo kim, the pennsylvania state university, 2018-2019"""


import recordlinkage, sys, pandas, numpy, scipy, networkx
 
def dedupe(bmatrix, features, df, e = .05):

    """
    Classify the data into the links and the non-links probabilistically.
    
    It uses Fellegi-Sunter algorithm. The frequencies of first and last names
    are weighted based on the Method I. For details, see Fellegi and Sunter
    (1969).
    
    Parameters
    ----------
    bmatrix : pandas.DataFrame
        Boolean matrix of feature comparison you want to use
    features : a list of three lists
        The first list should contain the feature set. The second one should
        contain the list of features you want to use as interaction terms. This
        list should be the slice position in ['firstname', 'lastname',
        'suffix'] + features[0]. The third one should contain the list of
        features you want to apply for Fellegi-Sunter frequency weights
        (Method I).
    df : pandas.DataFrame
        Data Frame you want to classify.
    e : float
        Overall error rate. Default is .05.

    Returns
    -------
    bmatrix : pandas.DataFrame
        Boolean matrix of feature comparison with a similarity score
    """
    
    #initiate
    epsilon = sys.float_info.epsilon        #prevent 0 or 1
    feat_list = ['firstname', 'lastname', 'suffix'] + features[0]
    vector = bmatrix.reset_index()
    
    #herzog et al. (2007: 90)
    var_order = 0
    for var in feat_list:
        
        print(var)
        vector = vector.join(df[var], on = 'level_0')
        vector.rename(columns = {var: '_f'}, inplace = True)
        vector = vector.join(df[var], on = 'level_1')
        vector.rename(columns = {var: '_g'}, inplace = True)
        
        #m-probability
        vector['_m'] = ((1 - e) * (1 - e)) + (e * e) - epsilon

        #u-probability
        vector['_u'] = (1 / df[var].nunique()) + epsilon

        #weights
        vector['_w_a'] = numpy.log2(vector['_m'] / vector['_u'])
        vector['_w_d'] = numpy.log2((1 - vector['_m']) / (1 - vector['_u']))

        #fellegi-sunter's method i
        if var in features[2]:
        
            #reset index
            freq = df[var].value_counts().reset_index()
            
            #join columns, level-f
            vector = vector.merge(freq, left_on = '_f', right_on = 'index')
            vector.rename(columns = {var: '_freq_f'}, inplace = True)
            vector.drop(columns = ['index'], inplace = True)
            
            #join columns, level-g
            vector = vector.merge(freq, left_on = '_g', right_on = 'index')
            vector.rename(columns = {var: '_freq_g'}, inplace = True)
            vector.drop(columns = ['index'], inplace = True)

            #large n
            vector['_fg'] = vector['_f'] + vector['_g']
            N = vector['_fg'].value_counts().reset_index()
            vector = vector.merge(N, how = 'left',
                                  left_on = '_fg', right_on = 'index')
            
            #agreement
            err1 = (1 - e) * (1 - e) * (1 - e) * (1 - e) * (1 - e)
            vector['min'] = numpy.where(vector['_freq_f'] < vector['_freq_g'],
                                        vector['_freq_f'], vector['_freq_g'])
            vector['_m_a'] = (vector['min'] / vector.shape[0]) * err1
            err2 = (1 - e - e - e - e - e)
            vector['_u_a'] = (vector['_freq_f'] / df[var].shape[0]) * \
                             (vector['_freq_g'] / df[var].shape[0]) * err2
            vector['_w_a'] = numpy.log2(vector['_m_a'] / vector['_u_a'])

            """
            #fellegi and sunter (1969: 1193)
            vector['_m_a'] = (vector['_fg_y'] / vector.shape[0]) * err1

            #herzog et al. 2007: 95)
            vector['_m_a'] = (vector['min'] / vector.shape[0]) * err1
            """
            
            #disagreement
            err3 = e + e + e
            vector['_m_d'] = err3
            err4 = (1 - e - e - e)
            prop = freq[var] / df[var].value_counts().sum()
            prop_sumsq = (prop ** 2).sum()
            vector['_u_d'] = 1 - (err4 * prop_sumsq)
            vector['_w_d'] = numpy.log2(vector['_m_d'] / vector['_u_d'])

            #cleanse vector
            coldrop = ['_m', '_u', '_freq_f', '_freq_g', '_fg_y', '_m_a',
                       '_u_a', '_m_d', '_u_d', '_fg_x', 'index']
            vector.drop(columns = coldrop, inplace = True)

        #assign proper value
        notna_cond = vector['_f'].notna() & vector['_g'].notna()
        v = vector[var_order] == 1
        proper_weight = numpy.where(v, vector['_w_a'], vector['_w_d'])
        vector['w_'+var] = numpy.where(notna_cond, proper_weight, 0)
        coldrop = ['_f', '_g', '_w_a', '_w_d']
        vector.drop(columns = coldrop, inplace = True)
        var_order = var_order + 1
    
    #interaction terms
    if len(features[1][0]) > 0:        #if any
        
        it_order = 0
        for it in features[1]:
            
            #concatenate multiple columns into one
            it_vars = [feat_list[i] for i in it]
            varname = 'w_it' + str(it_order)
            print(it_vars)
            
            #m-probability
            vector['_m'] = ((1 - e) * (1 - e)) + (e * e) - epsilon
    
            #u-probability
            df['_'] = df[it_vars].apply(lambda x: ''.join(x.astype(str)),
                                        axis = 1)
            df['_'] = numpy.where(df['_'].str.contains('nan'),
                                  numpy.nan, df['_'])
            vector['_u'] = (1 / df['_'].unique().size) + epsilon
    
            #compute weights
            vector['_w_a'] = numpy.log2(vector['_m'] / vector['_u'])
            vector['_w_d'] = numpy.log2((1 - vector['_m']) / (1 - vector['_u']))
            proper_weight = numpy.where(vector[it_order + var_order] == 1,
                                        vector['_w_a'], vector['_w_d'])
            
            #missing values
            ifany_miss = (vector[['w_' + i for i in it_vars]] == 0).any(axis=1)
            vector[varname] = numpy.where(ifany_miss, 0, proper_weight)
            it_order = it_order + 1
    
            #cleanse vector
            coldrop = [col for col in vector.columns if '_w_' in str(col)]
            vector.drop(columns = coldrop, inplace = True)
            df.drop(columns = '_', inplace = True)
    
    #summation
    tbs_fields = [col for col in vector.columns if 'w_' in str(col)]
    vector['fs_score'] = vector[tbs_fields].sum(axis = 1)
    vector.set_index(['level_0', 'level_1'], inplace = True)
    
    return vector







    
def match(bmatrix, features, *dframes, e = .05):

    """
    Classify the data into the links and the non-links probabilistically.
    
    It uses Fellegi-Sunter algorithm. The frequencies of first and last names
    are weighted based on the Method I. For details, see Fellegi and Sunter
    (1969).
    
    Parameters
    ----------
    bmatrix : pandas.DataFrame
        Boolean matrix of feature comparison you want to use
    features : a list of three lists
        The first list should contain the feature set. The second one should
        contain the list of features you want to use as interaction terms. This
        list should be the slice position in ['firstname', 'lastname',
        'suffix'] + features[0]. The third one should contain the list of
        features you want to apply for Fellegi-Sunter frequency weights
        (Method I).
    df : pandas.DataFrame
        Data Frame you want to classify.
    e : float
        Overall error rate. Default is .05.

    Returns
    -------
    bmatrix : pandas.DataFrame
        Boolean matrix of feature comparison with a similarity score
    """
    
    #initiate
    dfA, dfB = dframes[0], dframes[1]
    epsilon = sys.float_info.epsilon        #prevent 0 or 1
    feat_list = ['firstname', 'lastname', 'suffix'] + features[0]
    vector = bmatrix.reset_index()
    
    #herzog et al. (2007: 90)
    var_order = 0
    for var in feat_list:
        
        print(var)
        vector = vector.join(dfA[var], on = 'level_0')
        vector.rename(columns = {var: '_f'}, inplace = True)
        vector = vector.join(dfB[var], on = 'level_1')
        vector.rename(columns = {var: '_g'}, inplace = True)
        
        #m-probability
        vector['_m'] = ((1 - e) * (1 - e)) + (e * e) - epsilon

        #u-probability
        vector['_u'] = (1 / dfA[var].nunique()) + epsilon

        #weights
        vector['_w_a'] = numpy.log2(vector['_m'] / vector['_u'])
        vector['_w_d'] = numpy.log2((1 - vector['_m']) / (1 - vector['_u']))

        #fellegi-sunter algorithm should come here but we decided not to use it
        
        #assign proper value
        notna_cond = vector['_f'].notna() & vector['_g'].notna()
        v = vector[var_order] == 1
        proper_weight = numpy.where(v, vector['_w_a'], vector['_w_d'])
        vector['w_'+var] = numpy.where(notna_cond, proper_weight, 0)
        coldrop = ['_f', '_g', '_w_a', '_w_d']
        vector.drop(columns = coldrop, inplace = True)
        var_order = var_order + 1
        
    #interaction terms
    if len(features[1][0]) > 0:        #if any
        
        it_order = 0
        for it in features[1]:
            
            #concatenate multiple columns into one
            it_vars = [feat_list[i] for i in it]
            varname = 'w_it' + str(it_order)
            print(it_vars)
            
            #m-probability
            vector['_m'] = ((1 - e) * (1 - e)) + (e * e) - epsilon
    
            #u-probability
            dfA['_'] = dfA[it_vars].apply(lambda x: ''.join(x.astype(str)),
                                          axis = 1)
            dfA['_'] = numpy.where(dfA['_'].str.contains('nan'),
                                   numpy.nan, dfA['_'])
            vector['_u'] = (1 / dfA['_'].unique().size) + epsilon
    
            #compute weights
            vector['_w_a'] = numpy.log2(vector['_m'] / vector['_u'])
            vector['_w_d'] = numpy.log2((1 - vector['_m']) / (1 - vector['_u']))
            proper_weight = numpy.where(vector[it_order + var_order] == 1,
                                        vector['_w_a'], vector['_w_d'])
            
            #missing values
            ifany_miss = (vector[['w_' + i for i in it_vars]] == 0).any(axis = 1)
            vector[varname] = numpy.where(ifany_miss, 0, proper_weight)
            it_order = it_order + 1
    
            #cleanse vector
            coldrop = [col for col in vector.columns if '_w_' in str(col)]
            vector.drop(columns = coldrop, inplace = True)
            dfA.drop(columns = '_', inplace = True)

    #summation
    tbs_fields = [col for col in vector.columns if 'w_' in str(col)]
    vector['fs_score'] = vector[tbs_fields].sum(axis = 1)
    vector.set_index(['level_0', 'level_1'], inplace = True)
    
    return vector


    
    
    
    
    
    
    

    

def tiebreak(bmatrix, df):

    """
    Conduct post-hoc tie-breaking.
    
    There is a small proportion of ties that have fairly high similarity scores
    even though they are obviously non-links. In duplication, they are mainly
    siblings (twins, particularly). This process aims to break such ties with
    a couple of assumptions. GPS and CPS have different rules for tie-breaking.
    For details, see Technical Report.
    
    Parameters
    ----------
    bmatrix : pandas.DataFrame
        Boolean matrix of feature comparison you want to use
    df : pandas.DataFrame
        Data frame that contains features to be used for tie breaking.

    Returns
    -------
    bmatrix : pandas.DataFrame
        Boolean matrix of feature comparison excluding broken ties
    broken : pandas.DataFrame
        Boolean matrix of feature comparison in which ties are broken. Only for
        clerical review.
    """
    
    
    #prepare features for breaking ties
    bmatrix = bmatrix.reset_index()
    tbm = df[['referralid', 'firstname', 'lastname', 'dob', 'dt_referral',
              'county', 'referraltype']]
    bmatrix = bmatrix.merge(tbm, left_on = 'level_0', right_index = True)
    bmatrix = bmatrix.merge(tbm, left_on = 'level_1', right_index = True)    
    
    #conditions for gps
    gps0a = bmatrix['referraltype_x'] == bmatrix['referraltype_y']
    gps0b = bmatrix['referraltype_x'] == 'GPS'
    gps1 = bmatrix['referralid_x'] == bmatrix['referralid_y']
    gps2 = bmatrix['firstname_x'] != bmatrix['firstname_y']
    gps3 = bmatrix['lastname_x'] != bmatrix['lastname_y']
    gps4 = bmatrix['dob_x'] != bmatrix['dob_y']
    
    #conditions for cps
    cps0a = bmatrix['referraltype_x'] == bmatrix['referraltype_y']
    cps0b = bmatrix['referraltype_x'] == 'CPS'
    cps1a = bmatrix['dt_referral_x'] == bmatrix['dt_referral_y']
    cps1b = bmatrix['county_x'] == bmatrix['county_y']
    cps2 = bmatrix['firstname_x'] != bmatrix['firstname_y']
    cps3 = bmatrix['lastname_x'] != bmatrix['lastname_y']
    cps4 = bmatrix['dob_x'] != bmatrix['dob_y']
    
    #hic rhodus, hic salta
    broken_gps = bmatrix[(gps0a & gps0b) & gps1            & (gps2 | gps3)]
    broken_cps = bmatrix[(cps0a & cps0b) & (cps1a & cps1b) & (cps2 | cps3)]
    broken = broken_gps.append(broken_cps)
    combined = bmatrix.append(broken)
    combined = combined[~combined.index.duplicated(keep = False)]
    bmatrix = combined.set_index(['level_0', 'level_1'])
    
    return bmatrix, broken














def components(df, edges, newvar, merge_by, fill_by):
    
    """
    Identify social network components.
    
    Social network analysis provides useful tools for deduplication and
    matching. Component identification is particularly to identify unique
    components (including isolated nodes) and assign them unique ID numbers.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data frame in which component IDs will be assigned.
    edges : pandas.core.series.Series
        MultiIndex that representa a graph structure. Components will be
        identified from this MultiIndex.
    newvar : str
        Name of unique ID that represents components.
    merge_by : str
        The criterion that defines component relationship. You can give a
        variable name or 'index'.
    fill_by : str
        Isolated nodes (or archepelago) do not have component IDs. A series of
        unique IDs will be assigned by this variable. This variable should not
        have any missing values.

    Returns
    -------
    subset : pandas.DataFrame
        Data frame with component IDs
    """

    _temp = edges.reset_index()
    col0 = _temp.columns[0]
    col1 = _temp.columns[1]
    G = networkx.from_pandas_edgelist(_temp, col0, col1)
    family = [c for c in networkx.connected_components(G)]
    
    idx = 1
    fam = pandas.DataFrame()
    for i in family:
        tba = pandas.DataFrame(i)
        tba[newvar] = idx
        fam = fam.append(tba, ignore_index = True, sort = False)
        idx = idx + 1

    #merge to subset
    if merge_by == 'index':
        famid = df.merge(fam, how = "left", left_index = True, right_on = 0)
        
    else:
        famid = df.merge(fam, how = "left", left_on = merge_by, right_on = 0)
        
    famid['alt'] = famid.groupby(fill_by).grouper.group_info[0] + idx + 1000
    famid[newvar] = numpy.where(famid[newvar].isna(), famid['alt'],
                                     famid[newvar])
    famid[newvar] = famid[newvar].astype(int)
    famid.drop(columns = [0, 'alt'], inplace = True)

    return famid










    

def comembership(matrix):

    """
    Conduct matrix multiplication to generate a design matrix from a 2-mode
    affiliated network.
    
    Social network analysis provides useful tools for deduplication and
    matching. Co-membership identification can be used to identify a shared
    membership in a group such as a family.
    
    Parameters
    ----------
    matrix : pandas.DataFrame
        Data frame in which co-membership will be assigned.

    Returns
    -------
    _df : pandas.core.frame.DataFrame
        Data frame with co-membership dyads
    """
    
    #drop if any missing in the given matrix
    mcol = matrix.columns
    matrix = matrix.sort_values(mcol[1])
    matrix = matrix[(matrix[mcol[0]].notna()) & (matrix[mcol[1]].notna())]
    
    #prepare a sparse matrix of n-by-m
    index_to_mcol = pandas.DataFrame(matrix[mcol[1]].unique()).reset_index()
    matrix = matrix.drop_duplicates()
    matrix['value'] = 1
    matrix.set_index(mcol.tolist(), inplace = True)
    smat = scipy.sparse.coo_matrix((matrix['value'],
                                   (matrix.index.labels[0],
                                    matrix.index.labels[1])))
    
    #compute M'M and place it in df
    rmatrix = smat.T * smat
    x = [a for a in rmatrix.nonzero()[0]]
    y = [a for a in rmatrix.nonzero()[1]]
    indices = []
    for i in range(0, len(x)):       #len(x)==len(y)
        if x[i] != y[i]:
            indices.append([x[i], y[i]])
    _df = pandas.DataFrame(indices)
    _df['value'] = 1
    
    #assign the first column to the common key, level_0
    temp = _df.merge(index_to_mcol, how = "inner",
                     left_on = 0, right_on = 'index')
    temp = temp[['0_y', 1, 'value']]
    temp.rename(columns={'0_y': 'level_0'}, inplace=True)
    
    #assign the first column to the common key, level_1
    temp = temp.merge(index_to_mcol, how="inner", left_on=1,
                      right_on='index')
    _df = temp[['level_0', 0, 'value']]
    _df.rename(columns={0: 'level_1'}, inplace=True)
    #_df['level_0'] = _df['level_0'].astype(int)
    #_df['level_1'] = _df['level_1'].astype(int)
    
    #set index
    _df.set_index(['level_0', 'level_1'], inplace=True)

    return _df
    
     








