"""will be deleted sooner or later"""

    



def assign_ij(group, column):

    """
    For Stata-like easier reshapes
    
    Usage:
        
    text['i'], text['j'] = assign_ij(text, 'speaker')
    data = text.pivot_table(index = ['i', 'speaker', 'topic'],
                            columns = 'j',
                            values = 'talk',
                            fill_value = '',
                            aggfunc = lambda x:' '.join(x))
    """

    #index (i)
    indnum = 1
    i = [indnum]
    for rep1 in range(1, len(group)):
        if group[column].iloc[rep1] is not group[column].iloc[rep1-1]:
            indnum = indnum + 1
        if group[column].iloc[rep1]==group[column].iloc[rep1-1]:
            indnum = indnum
        i.append(indnum)
        
    #column (j)
    colnum = 1
    j = [colnum]
    for rep2 in range(1, len(group)):
        if group[column].iloc[rep2] is not group[column].iloc[rep2-1]:
            colnum = 1
        if group[column].iloc[rep2]==group[column].iloc[rep2-1]:
            colnum = colnum + 1
        j.append(colnum)

    return i, ['col'+str(t) for t in j]







def crossfield():
    
    """
    
    This demonstrate how cross-field indexing and comparison work.
    
    """
    
    df=pandas.DataFrame({'firstname': ['JOHN','JOHN','DOE', 'HYUN', 'KIM'],
                         'lastname': ['DOE', 'DOE', 'JOHN', 'KIM',  'HYUN']})
    
    #index 1
    indexer1 = recordlinkage.Index()
    indexer1.block(['firstname', 'lastname'])
    cpairs1 = indexer1.index(df)
    print(cpairs1)
    
    #index 2
    indexer2 = recordlinkage.Index()
    indexer2.block(left_on = ['firstname', 'lastname'],
                  right_on = ['lastname', 'firstname'])
    cpairs2 = indexer2.index(df)
    print(cpairs2)
    
    #pooled
    pooled = cpairs1.append(cpairs2)
    print([i for i in pooled])
    
    #comparison
    comparer = recordlinkage.Compare(n_jobs = 6)
    comparer.string('firstname', 'firstname',
                    method = 'jarowinkler', threshold = .9)
    comparer.string('lastname', 'lastname',
                    method = 'jarowinkler', threshold = .9)
    comparer.string('firstname', 'lastname',
                    method = 'jarowinkler', threshold = .9)
    comparer.string('lastname', 'firstname',
                    method = 'jarowinkler', threshold = .9)
    comparer.compute(pooled, df)
    
    








def fillna_id(subset, newvar, fill_by):
    
    #fill na with a sequence of unique ids
    tempid = subset[newvar].max().astype(int) + 1
    lowest = subset[fill_by].min().astype(int)
    subset['_tempid'] = subset[fill_by] + tempid - lowest
    subset[newvar].fillna(subset['_tempid'], inplace = True)
    subset[newvar] = subset[newvar].astype(int)
    subset.drop(columns=['_tempid'], inplace = True)

    return subset


