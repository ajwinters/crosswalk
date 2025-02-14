"""hyun woo kim, the pennsylvania state university, 2018-2019"""


import pandas, numpy, jellyfish, crosswalk.evaluation




def standardize(fields, df):

    """
    Standardize name and suffix, SSN, date and county fields of the given data.

    Various fields in administrative data should be standardized before
    conducting a probabilistic record linkage. Standardization process includes
    simplification and trimming of characters in the first and last name fields
    as well as handling missing values in other fields. Be sure that your data
    frame contains ``firstname'', ``lastname'', ``suffix'', ``mciid'', ``ssn'',
    ``county'', ``dobyy'', ``dobmm'', and ``dobdd''.
   
    
    Parameters
    ----------
    df : pandas.DataFrame
        A pandas.DataFrame you want to standardize.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame with an additional field 'valid' (1 indicates valid
        observation, 0 otherwise). 
    """
    
    #trim and transform to upper cases
    for i in [fields['firstname'], fields['lastname']]:
        df[i] = df[i].str.strip().str.upper()
        df[i] = df[i].replace("^'", "", regex=True)
        
    #multiple persons
    mpersons = " AND | BUT "
    df['valid'] = 1
    for i in [fields['firstname'], fields['firstname']]:
        df['valid'] = numpy.where(df[i].str.contains(mpersons), 0, df['valid'])
    
    #straightforward exceptions
    stexcept = {"&QUOTE;JUNIOR&QUOTE;": "", "JAME'S": "JAMES", "&#39;": "",
                "ISAIAHA'S": "ISAIAHAS", "JHON": "JOHN", "JONH": "JOHN"}
    df[fields['firstname']].replace(regex = stexcept, inplace = True)

    #weak first names, such as D J SMITH
    cond1 = df[fields['firstname']].str.len() <= 3
    cond2 = df[fields['firstname']].str.contains(" |\.|/")
    stronger = df[fields['firstname']].replace(regex = {' |\.': ''})
    df[fields['firstname']] = numpy.where(cond1 & cond2,
                                          stronger, df[fields['firstname']])
    df[fields['firstname']].replace(regex = {"MD": "MUHAMMAD"}, inplace = True)
    
    #end up with one whitespace and one letter in the first name
    whitespace = {" [\w{1}]$": "", " [\w{1}]\.": "", " [\w{1}] [\w{1}]$": "",
                  " [\w{1}] [\w{1}]\.$": "", " [\w{1}]\. [\w{1}]$": "",
                  " [\w{1}]\. [\w{1}]\.$": ""}
    df[fields['firstname']].replace(regex = whitespace, inplace = True)
    
    #drop if contains completely irrelevant
    agency1 = "CPS | CPS|^CPS$|CYF | CYF|^CYF$| AP |^AP$|^CYS$| CYS$|COCYS$|"
    agency2 = "DHS|HLTH| COUNTY|^COUNTY|PROVIDER"
    hopeless = agency1 + agency2
    for i in [fields['firstname'], fields['lastname']]:
        df[i].fillna('', inplace = True)
        df['valid'] = numpy.where(df[i].str.contains(hopeless), 0, df['valid'])
    df['valid'] = numpy.where((df[fields['firstname']] == "JOHN") &
                              (df[fields['lastname']] == "DOE"),
                              0, df['valid'])
    df['valid'] = numpy.where((df[fields['firstname']] == "JANE") &
                              (df[fields['lastname']] == "DOE"),
                              0, df['valid'])

    #general complete drop
    unknown = "^NOT$|^NOT |VAILABLE|NOTKNOWN|UNK |UNKNOWN|UNKN|UNK\.|^UNK$|^UKNO$| UNK|^UKN$|UNKWN|UNKNWN|UKNOWN|UNKWNON|UUNKNOWN|UNKOWN|NAMEUNK|UNKONWN|NAME$|UMKNOW|NAMELESS|NONAME|NO NAME|NOT GIVEN|NOT KNOW|NOTKNOW|ANONYMOUS|CASECHILD|CHILLD|^EXIST$|EXIST$| EXIST |EXISTED|NTKNWN|UNIKNOW|NOIDEA|^AKA$|PROVIDED|JOHNDOE|^NONE$|DUPLICATE|N/A|APPLICABLE|TERMINATED|"
    age = "Y/O|-Y-O| YO |-YROLD|YR\.|YR OLD|MO OLD|YO SON|YEAR OLD|YEARS OLD|MONTH OLD| GRADE |YRS OLD| YO | YO$|^YO$|ADULT|^BABY$| BABY|BABY |UNBORN|NEWBORN|CHILDREN|CHILLD|^CHILD$|CHILD |INFANT$|TODDLER|FETUS|UNBORN|DECEASE|XXX|"
    role = "^UNCLE$| UNCLE|^AUNT$| AUNT$|^WIFE|^FATHER$| FATHER$|^FATHER[ |/]|^MOTHER$| MOTHER$|^MOTHER[ |/]|GRANDMOTHER|DISCLOSED|PRISON|DAUGHTER|CHN$|^CH |CARETAKER|CAREGIVER|HHM|PARAMOUR$|TEENAGER|^TWIN$| TWIN|TWIN |JOHN-DOE|JANE-DOE|"
    gender = "FEMALE|^FEMALE$|MATERNAL|PATERNAL| GIRL|GIRL |^GIRL$| BOY$| BOY |BOYFRIEND| SHE | HER |^MR$|^MS$|^MRS$|"
    race = "AFRICAN AMERICAN|"
    other = "'S$| IS | ARE | NAME |^FIRST[ |/]|^SECOND[ |/]| FIRST$| SECOND$|^REPORT|REPORT$| REPORT |REPORTED|REPORTER|INFORMATION|VOID|EXIST|DELETE|^TPR$"
    stdname = unknown + age + role + gender + race + other
    for i in [fields['firstname'], fields['lastname']]:
        df['valid'] = numpy.where(df[i].str.contains(stdname) == True,
                                  0, df['valid'])

    #remove some parts
    role = "^SIR |MGM|MGF|MGFLG|MGDMA|PGM|PGF|FMOM| NM|WLG|"
    removal = "&#39;|&QUOT;|\`|\'|¿|Â|\+|\.|\#|\?|\*|\!|^-|[0-9]|PP#| PPN|^PRT$|^PR$|WFA|WMO$|NICKNAME|;|&AMP|~"
    for i in [fields['firstname'], fields['lastname']]:
        df[i] = df[i].str.replace(role + removal, '')

    #remove parenthesis properly
    for i in [fields['firstname'], fields['lastname']]:
        cond = df[i].str.contains('^\(') | df[i].str.contains('\)$')
        df[i] = numpy.where(cond,
                            df[i].replace('\(|\)', '', regex = True),     #full
                            df[i].replace('\([^)]*\)|\((.*)', '', regex = True)) #partial or unbalanced

    #split by slash or a.k.a
    split = "/| AKA |PARAS | OR "
    for i in [fields['firstname'], fields['lastname']]:
        df[i] = df[i].str.split(split, expand = True)[0]

    #hyphen, whitespace, and underscore
    spc2 = "-| |_"
    for i in [fields['firstname'], fields['lastname']]:
        df[i] = df[i].str.replace(spc2, '')

    #if less than 2 letters
    for i in [fields['firstname'], fields['lastname']]:
        df['valid'] = numpy.where(df[i].str.len() < 2, 0, df['valid'])
    
    #common components in first and last names
    common = "TEST"
    cond1 = df[fields['firstname']].str.contains(common)
    cond2 = df[fields['lastname']].str.contains(common)
    df['valid'] = numpy.where(cond1 & cond2, 0, df['valid'])

    #firstname and lastname are identical
    #df['valid'] = numpy.where(df['firstname']==df['lastname'], 0, df['valid'])


    #other than names (contains hyphen, underscore, irrelevant things)
    for i in ['ssn', 'suffix', 'mciid']:
        
        if fields.get(i) != None:
            
            df[fields[i]] = df[fields[i]].astype(str)
            cor = "^0$|^999999999$"
            df[fields[i]] = df[fields[i]].replace(cor, numpy.NaN, regex = True)
            df[fields[i]] = pandas.to_numeric(df[fields[i]], errors = 'coerce')
    
        #county
        if fields.get('county') != None:
            
            df[fields['county']].replace('', numpy.nan, inplace = True)
        
        #make date format consistent
        cond1 = fields.get('dobyy') != None
        cond2 = fields.get('dobmm') != None
        cond3 = fields.get('dobdd') != None
        if cond1 & cond2 & cond3:
        
            df[fields['dobyy']] = df[fields['dobyy']].astype(float)
            df[fields['dobmm']] = df[fields['dobmm']].astype(float)
            df[fields['dobdd']] = df[fields['dobdd']].astype(float)

    return df











def population_split(df, role_child = 'role_child', age = 'age', union = True):
    
    """
    Split the given data into two--(1) underage population and (2) those not.
    
    It uses two rules to identify underage population--a dummy variable
    indicating underage population ('role_child') and the age variable ('age').
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data set you want to split.
    role_child : str (optional)
        The name of field indicating underage population.
    age : str (optional)
        The name or field indicating age.
    union : boolean (optional)
        True if you want to use OR (union) condition. False if you want to
        use AND (intersection) condition. By default, underage population is
        identified with the condition, role_child == 1 OR age <= 18.

    Returns
    -------
    majors : pandas.DataFrame
        A pandas DataFrame containing those who are not underage population.
    minors : pandas.DataFrame        
        A pandas DataFrame containing underage population.
    """
    
    
    #(1) role is child
    df['_minor1'] = numpy.where(df['role_child'] == 1, 1, 0)

    #(2) under 18
    df['age'] = (df['dt_referral'] - df['dob']) / 365.25
    df['age'] = numpy.where(df['age'] < 0, 0, df['age'])
    df['_minor2'] = numpy.where(df['age'] <= 18, 1, 0)

    #extract minors as a subset of df
    if union == True:
        df['_minor'] = numpy.where((df['_minor1'] == 1) |
                                   (df['_minor2'] == 1), 1, 0)
    else:
        df['_minor'] = numpy.where((df['_minor1'] == 1) &
                                   (df['_minor2'] == 1), 1, 0)
    df.drop(columns = ['_minor1', '_minor2'], inplace = True)
    
    #minors
    minors = df[df['_minor'] == 1]
    minors.drop(columns = ['_minor'], inplace = True)

    #majors
    majors = df[(df['_minor'] == 0)]
    majors.drop(columns = ['_minor'], inplace = True)    

    #majors.shape[0] + minors.shape[0] == CWIS.shape[0]
    if majors.shape[0] + minors.shape[0] != df.shape[0]:
        raise ValueError("Unconformable matrices.")
    

    
    return majors, minors









    
    
    
def sarahs_rule(majors, minors, REL):

    """
    Apply Sarah's Rule for flat data to obtain dyadic data
    
    Sarah's Rule is useful to transform flat data--each observation represents
    individual person with a field indicating his or her relationship to others
    to dyadic data--each observation represents two individual persons with a
    field indicating their relationship.
    
    For CWIS, I used to only use Relationship file to generate the dyad records
    of minor-to-biomom. Later, Sarah Font suggested using additional rule to
    identify biological mothers. This Rule does not override information of the
    Relationship file. For details, see Technical Report.
    
    Parameters
    ----------
    majors : pandas.DataFrame
        A pandas DataFrame containing those who are not underage population.
    minors : pandas.DataFrame        
        A pandas DataFrame containing underage population.
    REL : pandas.DataFrame
        Preexisting dyadic data

    Returns
    -------
    REL : pandas.DataFrame
        Preexisting dyadic data PLUS new dyadic relationship from flat data
    """
    
    fields = ['referralid', 'referralpersonid', 'age', 'firstname', 'lastname',
              'female', 'role_parent']
    newrel = minors[fields].merge(majors[fields], how = "outer",
                                  right_on = 'referralid',
                                  left_on = 'referralid')
    newrel = newrel[newrel.referralpersonid_x.notna()]
    
    #sarah's rule 1
    newrel = newrel[(newrel['role_parent_y']==1) & (newrel['female_y']==1)]
    rule1 = newrel.drop_duplicates(['referralid', 'referralpersonid_x'],
                                   keep = False)
    rule1 = rule1[['referralid', 'referralpersonid_x', 'referralpersonid_y']]
    rule1 = rule1.rename(columns = {'referralpersonid_x': 'allegedvictimid',
                                    'referralpersonid_y': 'referralpersonid'})
    rule1['relationship'] = 'Mother-Biological'
    
    #sarah's rule 2
    newrel = newrel[newrel.duplicated(['referralid', 'referralpersonid_x'],
                                     keep = False)]
    newrel = newrel[newrel['age_y'] - newrel['age_x'] >= 14]
    rule2 = newrel.drop_duplicates(['referralid', 'referralpersonid_x'],
                                   keep = False)
    rule2 = rule2[['referralid', 'referralpersonid_x', 'referralpersonid_y']]
    rule2 = rule2.rename(columns = {'referralpersonid_x': 'allegedvictimid',
                                    'referralpersonid_y': 'referralpersonid'})
    rule2['relationship'] = 'Mother-Biological'
    
    #sarah's rule 3
    newrel = newrel[newrel['lastname_x'] == newrel['lastname_y']]
    rule3 = newrel.drop_duplicates(['referralid', 'referralpersonid_x'],
                                   keep = False)
    rule3 = rule3[['referralid', 'referralpersonid_x', 'referralpersonid_y']]
    rule3 = rule3.rename(columns = {'referralpersonid_x': 'allegedvictimid',
                                    'referralpersonid_y': 'referralpersonid'})
    rule3['relationship'] = 'Mother-Biological'
    
    #venn diagram evaluation
    biomom_rel = REL[REL['relationship'] == 'Mother-Biological']
    venn_diagram(biomom_rel, [rule1, rule2, rule3])
    
    #prepare the return
    REL = REL.append(rule1, ignore_index = True, sort = False)
    REL = REL.append(rule2, ignore_index = True, sort = False)
    REL = REL.append(rule3, ignore_index = True, sort = False)
    REL.drop_duplicates(['referralid', 'allegedvictimid', 'referralpersonid'],
                        inplace = True, keep = 'first')
    
    return REL

    









def venn_diagram(rfile, rules):

    """
    For internal use.
    """
    
    _tbc = rules[0].append(rules[1], ignore_index = True, sort = False)
    _tbc = rules[0].append(rules[2], ignore_index = True, sort = False)
    _tbc.drop_duplicates(['referralid', 'allegedvictimid', 'referralpersonid'],
                        inplace = True, keep = 'first')
    
    #mommy AND (rule1 & rule2 & rule3)
    _a = pandas.concat([rfile, _tbc])
    intersection = _a[_a.duplicated(['referralid',
                                     'allegedvictimid',
                                     'referralpersonid',
                                     'relationship'], keep = 'first')]
    result1 = len(intersection)
    
    #mommy BUT ~(rule1 & rule2 & rule3)
    result2 = len(rfile) - len(intersection)

    #~mommy BUT (rule1 & rule2 & rule3)
    result3 = len(_tbc) - len(intersection)
    
    return [len(rfile), len(_tbc), result1, result2, result3]
    
    
    




    
def biomoms(majors, minors, REL):

    """
    Extract biological mothers from the Relationship file
    
    Generates the list of biological mothers that will should be deduplicated
    subsequently. The list also contains useful features for deduplication.
    
    Parameters
    ----------
    majors : pandas.DataFrame
        A pandas DataFrame containing those who are not underage population.
    minors : pandas.DataFrame        
        A pandas DataFrame containing underage population.
    REL : pandas.DataFrame
        The Relationship file

    Returns
    -------
    biomom : pandas.DataFrame
        Extracted biological mothers
    """
    
    #moms's info
    moms_info = majors[(majors['female'] == 1)]
    
    #merge moms' info to relationship file
    mnk = REL[REL['relationship'] == "Mother-Biological"]
    mnk = mnk.merge(moms_info, how = 'left',
                    left_on = ['referralid', 'referralpersonid'],
                    right_on = ['referralid', 'referralpersonid'])
    
    #expunged or 404
    mnk = mnk[mnk['personid'].notna()]           #len(mnk)=469,952

    #merge kid's info to relationship
    kids_info = minors[['referralid', 'referralpersonid', 'firstname',
                        'lastname', 'dob', 'ssn', 'mciid']]
    mnk = mnk.merge(kids_info, how = 'left',
                    left_on = ['referralid', 'allegedvictimid'],
                    right_on = ['referralid', 'referralpersonid'])
    mnk.rename(columns = {'referralpersonid_x': 'referralpersonid',
                          'firstname_x': 'firstname',
                          'lastname_x': 'lastname',
                          'dob_x': 'dob',
                          'mciid_x': 'mciid',
                          'ssn_x': 'ssn',
                          'firstname_y': 'kidfname',
                          'lastname_y': 'kidlname',
                          'dob_y': 'kid_dob',
                          'ssn_y': 'kid_ssn',
                          'mciid_y': 'kid_mciid'}, inplace = True)

    #kid's name    
    for i in ['kidfname', 'kidlname']:
        mnk[i].fillna('', inplace = True)
        mnk[i] = mnk[i].apply(jellyfish.soundex)
    mnk['kid_name'] = mnk['kidfname'] + mnk['kidlname']
    mnk['kid_name'].replace('', numpy.nan, inplace = True)
    
    #prepare to go home
    dropcol = ['referralpersonid_y', 'kidlname', 'kidfname', 'relationship',
               'role_perp', 'role_child', 'role_parent', 'role_caretaker',
               'role_guardian', 'role_reporter', 'role_collat', 'nroles']
    biomom = mnk.drop(columns = dropcol)
    
    return biomom

    
    
    
    
    


    
    
def children(majors, minors, REL):
    
    """
    Extract children from the Relationship file

    Generates the list of children that will should be deduplicated
    subsequently. The list also contains useful features for deduplication. A
    couple of these features are graph measures. Thus, we need a priority rule.
    For mother, it is bio - adoptive - step (or unknown). For father, bio -
    adoptive (or legal) - step (or unknown). For example, if Jane Doe has no
    bio mother, but an adoptive mother and a step-mother, adoptive mother's
    name will be used.

    Parameters
    ----------
    majors : pandas.DataFrame
        A pandas DataFrame containing those who are not underage population.
    minors : pandas.DataFrame        
        A pandas DataFrame containing underage population.
    REL : pandas.DataFrame
        The Relationship file.

    Returns
    -------
    child : pandas.DataFrame
        Extracted children.
    """

    
    #major as others
    other = majors[['referralid', 'referralpersonid', 'firstname', 'lastname',
                    'dob', 'ssn', 'mciid']]
    
    #choose best moms
    REL['prior'] = 0
    REL['prior'] = numpy.where(REL['relationship'].str.contains('Mother'),
                               3, REL['prior'])
    REL['prior'] = numpy.where(REL['relationship']=='Mother-Adoptive',
                               2, REL['prior'])
    REL['prior'] = numpy.where(REL['relationship']=='Mother-Biological',
                               1, REL['prior'])
    moms = REL[(REL['prior'] != 0)]
    moms.sort_values(by = ['referralid', 'allegedvictimid', 'prior'],
                     inplace = True)
    moms.drop_duplicates(subset=['referralid', 'allegedvictimid'],
                         keep = 'first', inplace = True)
    
    #choose best dads
    REL['prior'] = 0
    REL['prior'] = numpy.where((REL['relationship'].str.contains('Father')),
                                3, REL['prior'])
    REL['prior'] = numpy.where((REL['relationship'] == 'Father-Adoptive') |
                                (REL['relationship'] == 'Father-Legal'), 
                               2, REL['prior'])
    REL['prior'] = numpy.where(REL['relationship'] == 'Father-Biological',
                                1, REL['prior'])
    dads = REL[(REL['prior'] != 0)]
    dads.sort_values(by = ['referralid', 'allegedvictimid', 'prior'],
                     inplace = True)
    dads.drop_duplicates(subset = ['referralid', 'allegedvictimid'],
                         keep = 'first', inplace = True)

    #mom's name SDX
    moms_rel = moms.merge(other, how = 'left',
                          left_on = ['referralid', 'referralpersonid'],
                          right_on = ['referralid', 'referralpersonid'])
    moms_rel.drop(columns = ['relationship', 'prior'], inplace = True)
    for name in ['firstname', 'lastname']:
        moms_rel[name].fillna('', inplace = True)
    moms_rel['SDX'] = moms_rel['firstname'].apply(jellyfish.soundex) + \
                      moms_rel['lastname'].apply(jellyfish.soundex)

    #dad's name SDX
    dads_rel = dads.merge(other, how = 'left',
                          left_on = ['referralid', 'referralpersonid'],
                          right_on = ['referralid', 'referralpersonid'])
    dads_rel.drop(columns = ['relationship', 'prior'], inplace = True)
    for name in ['firstname', 'lastname']:
        dads_rel[name].fillna('', inplace = True)
    dads_rel['SDX'] = dads_rel['firstname'].apply(jellyfish.soundex) + \
                      dads_rel['lastname'].apply(jellyfish.soundex)
    
    #merges dad's info
    family = minors.merge(moms_rel, how = 'left',
                          left_on = ['referralid', 'referralpersonid'],
                          right_on = ['referralid', 'allegedvictimid'])
    family.drop(columns = ['allegedvictimid', 'referralpersonid_y',
                           'firstname_y', 'lastname_y'], inplace = True)
    family.rename(columns = {'referralpersonid_x': 'referralpersonid',
                             'firstname_x': 'firstname',
                             'lastname_x': 'lastname',
                             'SDX': 'SDX_mom',
                             'dob_x': 'dob',
                             'ssn_x': 'ssn',
                             'mciid_x': 'mciid',
                             'dob_y': 'dob_mom',
                             'ssn_y': 'ssn_mom',
                             'mciid_y': 'mci_mom'}, inplace = True)

    #merges mom's info        
    family = family.merge(dads_rel, how = 'left',
                          left_on = ['referralid', 'referralpersonid'],
                          right_on = ['referralid', 'allegedvictimid'])
    family.drop(columns = ['allegedvictimid', 'referralpersonid_y', 'nroles',
                           'firstname_y', 'lastname_y'], inplace = True)
    family.rename(columns = {'referralpersonid_x': 'referralpersonid',
                           'firstname_x': 'firstname',
                           'lastname_x': 'lastname',
                           'SDX': 'SDX_dad',
                           'dob_x': 'dob',
                           'ssn_x': 'ssn',
                           'mciid_x': 'mciid',
                           'dob_y': 'dob_dad',
                           'ssn_y': 'ssn_dad',
                           'mciid_y': 'mci_dad'}, inplace = True)

    #prepare to go home       
    REL.drop(columns = ['prior'], inplace = True)
    for i in ['SDX_mom', 'SDX_dad']:
        family[i] = family[i].replace('', numpy.NaN)
    dropcol = ['role_perp', 'role_child', 'role_parent', 'role_caretaker',
           'role_guardian', 'role_reporter', 'role_collat']
    child = family.drop(columns = dropcol)
    
    return child

