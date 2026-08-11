"""hyun woo kim, the pennsylvania state university, 2018-2019"""


import pandas, numpy, jellyfish


def standardize(fields, df):
    """
    Standardize name and suffix, ID numbers, date and region fields of the given data.

    Various fields in administrative data should be standardized before
    conducting a probabilistic record linkage. Standardization process includes
    simplification and trimming of characters in the first and last name fields
    as well as handling missing values in other fields. Be sure that your data
    frame contains ``firstname'', ``lastname'', ``suffix'', ``pid2'', ``pid1'',
    ``region'', ``dobyy'', ``dobmm'', and ``dobdd''.

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

    # Trim and transform to upper cases
    for i in [fields['firstname'], fields['lastname']]:
        df[i] = df[i].str.strip().str.upper().replace("^'", "", regex=True)

    # Multiple persons
    mpersons = " AND | BUT "
    df['valid'] = 1
    for i in [fields['firstname'], fields['lastname']]:
        df['valid'] = numpy.where(df[i].str.contains(mpersons), 0, df['valid'])

    # Straightforward exceptions
    stexcept = {"&QUOTE;JUNIOR&QUOTE;": "", "JAME'S": "JAMES", "&#39;": "",
                "ISAIAHA'S": "ISAIAHAS", "JHON": "JOHN", "JONH": "JOHN"}
    df[fields['firstname']] = df[fields['firstname']].replace(regex=stexcept)

    # Weak first names, such as D J SMITH
    cond1 = df[fields['firstname']].str.len() <= 3
    cond2 = df[fields['firstname']].str.contains(" |\.|/")
    stronger = df[fields['firstname']].replace(regex={' |\.': ''})
    df[fields['firstname']] = numpy.where(cond1 & cond2, stronger, df[fields['firstname']])
    df[fields['firstname']] = df[fields['firstname']].replace(regex={"MD": "MUHAMMAD"})

    # End up with one whitespace and one letter in the first name
    whitespace = {" [\w{1}]$": "", " [\w{1}]\.": "", " [\w{1}] [\w{1}]$": "",
                  " [\w{1}] [\w{1}]\.$": "", " [\w{1}]\. [\w{1}]$": "",
                  " [\w{1}]\. [\w{1}]\.$": ""}
    df[fields['firstname']] = df[fields['firstname']].replace(regex=whitespace)

    # Drop unidentified-person placeholders
    for i in [fields['firstname'], fields['lastname']]:
        df[i] = df[i].fillna('')
    df['valid'] = numpy.where((df[fields['firstname']] == "JOHN") & (df[fields['lastname']] == "DOE"), 0, df['valid'])
    df['valid'] = numpy.where((df[fields['firstname']] == "JANE") & (df[fields['lastname']] == "DOE"), 0, df['valid'])

    # General complete drop
    unknown = "^NOT$|^NOT |VAILABLE|NOTKNOWN|UNK |UNKNOWN|UNKN|UNK\.|^UNK$|^UKNO$| UNK|^UKN$|UNKWN|UNKNWN|UKNOWN|UNKWNON|UUNKNOWN|UNKOWN|NAMEUNK|UNKONWN|NAME$|UMKNOW|NAMELESS|NONAME|NO NAME|NOT GIVEN|NOT KNOW|NOTKNOW|ANONYMOUS|^EXIST$|EXIST$| EXIST |EXISTED|NTKNWN|UNIKNOW|NOIDEA|^AKA$|PROVIDED|JOHNDOE|^NONE$|DUPLICATE|N/A|APPLICABLE|"
    age = "Y/O|-Y-O| YO |-YROLD|YR\.|YR OLD|YEAR OLD|YEARS OLD| GRADE |YRS OLD| YO | YO$|^YO$|ADULT|DECEASE|XXX|"
    gender = "FEMALE|^FEMALE$|MATERNAL|PATERNAL| GIRL|GIRL |^GIRL$| BOY$| BOY |BOYFRIEND| SHE | HER |^MR$|^MS$|^MRS$|"
    other = "'S$| IS | ARE | NAME |^FIRST[ |/]|^SECOND[ |/]| FIRST$| SECOND$|INFORMATION|VOID|EXIST|DELETE"
    stdname = unknown + age + gender + other
    for i in [fields['firstname'], fields['lastname']]:
        df['valid'] = numpy.where(df[i].str.contains(stdname), 0, df['valid'])

    # Remove some parts
    removal = "&#39;|&QUOT;|\`|\'|¿|Â|\+|\.|\#|\?|\*|\!|^-|[0-9]|PP#| PPN|^PRT$|^PR$|WFA|WMO$|NICKNAME|;|&AMP|~"
    for i in [fields['firstname'], fields['lastname']]:
        df[i] = df[i].str.replace(removal, '', regex=True)

    # Remove parenthesis properly
    for i in [fields['firstname'], fields['lastname']]:
        cond = df[i].str.contains('^\(') | df[i].str.contains('\)$')
        df[i] = numpy.where(cond, df[i].replace('\(|\)', '', regex=True), df[i].replace('\([^)]*\)|\((.*)', '', regex=True))

    # Split by slash or a.k.a
    split = "/| AKA |PARAS | OR "
    for i in [fields['firstname'], fields['lastname']]:
        df[i] = df[i].str.split(split, expand=True)[0]

    # Hyphen, whitespace, and underscore
    spc2 = "-| |_"
    for i in [fields['firstname'], fields['lastname']]:
        df[i] = df[i].str.replace(spc2, '', regex=True)

    # If less than 2 letters
    for i in [fields['firstname'], fields['lastname']]:
        df['valid'] = numpy.where(df[i].str.len() < 2, 0, df['valid'])

    # Common components in first and last names
    common = "TEST"
    cond1 = df[fields['firstname']].str.contains(common)
    cond2 = df[fields['lastname']].str.contains(common)
    df['valid'] = numpy.where(cond1 & cond2, 0, df['valid'])

    # Other than names (contains hyphen, underscore, irrelevant things)
    for i in ['pid1', 'suffix', 'pid2']:
        if fields.get(i) is not None:
            df[fields[i]] = df[fields[i]].astype(str).replace("^0$|^999999999$", numpy.nan, regex=True)
            df[fields[i]] = pandas.to_numeric(df[fields[i]], errors='coerce')

    # Region
    if fields.get('region') is not None:
        df[fields['region']] = df[fields['region']].replace('', numpy.nan)

    # Make date format consistent
    if fields.get('dobyy') and fields.get('dobmm') and fields.get('dobdd'):
        df[fields['dobyy']] = df[fields['dobyy']].astype(float)
        df[fields['dobmm']] = df[fields['dobmm']].astype(float)
        df[fields['dobdd']] = df[fields['dobdd']].astype(float)

    return df
