"""
Parameterized synthetic data generator for crosswalk record linkage testing.

Usage:
    from generate_data import generate_dedupe_data, generate_match_data

    df = generate_dedupe_data(n_records=10000, dedupe_rate=0.30, seed=42)
    dfA, dfB = generate_match_data(n_records=10000, match_rate=0.85, seed=42)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from name_variants import nickname_variant, phonetic_variant, ocr_variant


# ============================================================================
# Name pools (US Census top names — enough variety for realistic blocking)
# ============================================================================

FIRST_NAMES = [
    "JAMES", "MARY", "ROBERT", "PATRICIA", "JOHN", "JENNIFER", "MICHAEL",
    "LINDA", "DAVID", "ELIZABETH", "WILLIAM", "BARBARA", "RICHARD", "SUSAN",
    "JOSEPH", "JESSICA", "THOMAS", "SARAH", "CHRISTOPHER", "KAREN", "CHARLES",
    "LISA", "DANIEL", "NANCY", "MATTHEW", "BETTY", "ANTHONY", "MARGARET",
    "MARK", "SANDRA", "DONALD", "ASHLEY", "STEVEN", "KIMBERLY", "PAUL",
    "EMILY", "ANDREW", "DONNA", "JOSHUA", "MICHELLE", "KENNETH", "DOROTHY",
    "KEVIN", "CAROL", "BRIAN", "AMANDA", "GEORGE", "MELISSA", "TIMOTHY",
    "DEBORAH", "RONALD", "STEPHANIE", "EDWARD", "REBECCA", "JASON", "SHARON",
    "JEFFREY", "LAURA", "RYAN", "CYNTHIA", "JACOB", "KATHLEEN", "GARY",
    "AMY", "NICHOLAS", "ANGELA", "ERIC", "SHIRLEY", "JONATHAN", "ANNA",
    "STEPHEN", "BRENDA", "LARRY", "PAMELA", "JUSTIN", "EMMA", "SCOTT",
    "NICOLE", "BRANDON", "HELEN", "BENJAMIN", "SAMANTHA", "SAMUEL", "KATHERINE",
    "RAYMOND", "CHRISTINE", "GREGORY", "DEBRA", "FRANK", "RACHEL", "ALEXANDER",
    "CAROLYN", "PATRICK", "JANET", "JACK", "CATHERINE", "DENNIS", "MARIA",
    "JERRY", "HEATHER", "TYLER", "DIANE", "AARON", "RUTH", "JOSE",
    "JULIE", "ADAM", "OLIVIA", "NATHAN", "JOYCE", "HENRY", "VIRGINIA",
    "PETER", "VICTORIA", "ZACHARY", "KELLY", "DOUGLAS", "LAUREN", "HAROLD",
    "CHRISTINA", "CARL", "JOAN", "ARTHUR", "EVELYN", "GERALD", "JUDITH",
    "ROGER", "MEGAN", "KEITH", "ANDREA", "JEREMY", "CHERYL", "TERRY",
    "HANNAH", "LAWRENCE", "JACQUELINE", "SEAN", "MARTHA", "ALBERT", "GLORIA",
    "AUSTIN", "TERESA", "JOE", "ANN", "JESSE", "SARA", "WILLIE", "MADISON",
    "BILLY", "FRANCES", "BRUCE", "KATHRYN", "RALPH", "JANICE", "GABRIEL",
    "JEAN", "ROY", "ABIGAIL", "EUGENE", "ALICE", "RANDY", "JUDY", "WAYNE",
    "SOPHIA", "PHILIP", "GRACE", "RUSSELL", "DENISE", "BOBBY", "AMBER",
    "VINCENT", "DORIS", "LOUIS", "MARILYN", "HOWARD", "DANIELLE", "FRED",
    "BEVERLY", "JOHNNY", "ISABELLA", "ALAN", "THERESA", "JIMMY", "DIANA",
    "MARTIN", "NATALIE", "RAY", "BRITTANY", "CARLOS", "CHARLOTTE", "IVAN",
    "MARIE", "OSCAR", "KAYLA", "CRAIG", "ALEXIS", "CLARENCE", "LORI",
]

LAST_NAMES = [
    "SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES", "GARCIA", "MILLER",
    "DAVIS", "RODRIGUEZ", "MARTINEZ", "HERNANDEZ", "LOPEZ", "GONZALEZ",
    "WILSON", "ANDERSON", "THOMAS", "TAYLOR", "MOORE", "JACKSON", "MARTIN",
    "LEE", "PEREZ", "THOMPSON", "WHITE", "HARRIS", "SANCHEZ", "CLARK",
    "RAMIREZ", "LEWIS", "ROBINSON", "WALKER", "YOUNG", "ALLEN", "KING",
    "WRIGHT", "SCOTT", "TORRES", "NGUYEN", "HILL", "FLORES", "GREEN",
    "ADAMS", "NELSON", "BAKER", "HALL", "RIVERA", "CAMPBELL", "MITCHELL",
    "CARTER", "ROBERTS", "GOMEZ", "PHILLIPS", "EVANS", "TURNER", "DIAZ",
    "PARKER", "CRUZ", "EDWARDS", "COLLINS", "REYES", "STEWART", "MORRIS",
    "MORALES", "MURPHY", "COOK", "ROGERS", "GUTIERREZ", "ORTIZ", "MORGAN",
    "COOPER", "PETERSON", "BAILEY", "REED", "KELLY", "HOWARD", "RAMOS",
    "KIM", "COX", "WARD", "RICHARDSON", "WATSON", "BROOKS", "CHAVEZ",
    "WOOD", "JAMES", "BENNETT", "GRAY", "MENDOZA", "RUIZ", "HUGHES",
    "PRICE", "ALVAREZ", "CASTILLO", "SANDERS", "PATEL", "MYERS", "LONG",
    "ROSS", "FOSTER", "JIMENEZ", "POWELL", "JENKINS", "PERRY", "RUSSELL",
    "SULLIVAN", "BELL", "COLEMAN", "BUTLER", "HENDERSON", "BARNES", "GONZALES",
    "FISHER", "VASQUEZ", "SIMMONS", "GRAHAM", "MURRAY", "FORD", "CASTRO",
    "MARSHALL", "OWENS", "HARRISON", "FERNANDEZ", "MCDONALD", "WOODS",
    "WASHINGTON", "KENNEDY", "WELLS", "VARGAS", "HENRY", "CHEN", "FREEMAN",
    "WEBB", "TUCKER", "GUZMAN", "BURNS", "CRAWFORD", "OLSON", "SIMPSON",
    "PORTER", "HUNTER", "GORDON", "MENDEZ", "SILVA", "SHAW", "SNYDER",
    "MASON", "DIXON", "MUNOZ", "HUNT", "HICKS", "HOLMES", "PALMER",
    "WAGNER", "BLACK", "ROBERTSON", "BOYD", "ROSE", "STONE", "SALAZAR",
    "FOX", "WARREN", "MILLS", "MEYER", "RICE", "SCHMIDT", "GARZA",
    "DANIELS", "FERGUSON", "NICHOLS", "STEPHENS", "SOTO", "WEAVER", "RYAN",
    "GARDNER", "PAYNE", "GRANT", "DUNN", "KELLEY", "SPENCER", "HAWKINS",
    "ARNOLD", "PIERCE", "VAZQUEZ", "HANSEN", "PETERS", "SANTOS", "HART",
    "BRADLEY", "KNIGHT", "ELLIOTT", "CUNNINGHAM", "DUNCAN", "ARMSTRONG",
    "HUDSON", "CARROLL", "LANE", "RILEY", "ANDREWS", "ALVARADO", "RAY",
    "DELGADO", "BERRY", "PERKINS", "HOFFMAN", "JOHNSTON", "MATTHEWS",
    "MEDINA", "FOWLER", "BREWER", "HOFFMAN", "CARLSON", "SILVA", "PEARSON",
    "HOLLAND", "DOUGLAS", "FLEMING", "JENSEN", "BURTON", "ELLIS", "HARPER",
    "GEORGE", "LYNCH", "DEAN", "GILBERT", "GARRETT", "ROMERO", "WELCH",
    "LARSON", "FRAZIER", "BURKE", "HANSON", "MENDOZA", "MORENO", "BOWMAN",
    "WATKINS", "BARKER", "TERRY", "HOWE", "HAYNES", "LYONS", "GRAVES",
    "MILES", "PARK", "BLAKE", "NORRIS", "WATTS", "MCGUIRE", "RHODES",
    "MAXWELL", "WISE", "HARDY", "CROSS", "NASH", "WOLFE", "POTTER",
]

# ----------------------------------------------------------------------------
# Additional names spanning a broader range of origins (East/South Asian, Arabic
# / Middle Eastern, African, and more Hispanic/Latino). The matching pipeline is
# name-agnostic, but the original pools skewed Anglo/Western; these reduce that
# bias so the test data is more representative. They are interleaved across the
# rank spectrum (see _interleave) rather than appended to the low-frequency tail.
# ----------------------------------------------------------------------------

_DIVERSE_FIRST_NAMES = [
    # East Asian
    "WEI", "MING", "JING", "YAN", "HUI", "FENG", "HAO", "YONG", "HONG", "LEI",
    "MINJUN", "SEOJUN", "JIWOO", "SEOYEON", "JIMIN", "SOOJIN", "MINSEO",
    "ANH", "MINH", "HUONG", "THUY", "HOANG", "TRANG", "LINH",
    # South Asian
    "AARAV", "ARJUN", "ROHAN", "RAHUL", "AMIT", "RAVI", "SANJAY", "DEEPAK",
    "PRIYA", "ANANYA", "DIYA", "SAANVI", "KAVYA", "NEHA", "POOJA", "MEERA",
    # Arabic / Middle Eastern
    "MOHAMMED", "AHMED", "ALI", "OMAR", "HASSAN", "IBRAHIM", "YUSUF", "KHALID",
    "TARIQ", "SAMIR", "FATIMA", "AISHA", "LAYLA", "MARYAM", "NADIA", "AMINA",
    # African / African-American
    "DESHAWN", "JAMAL", "MALIK", "DARNELL", "TREVON", "DEANDRE", "MARQUIS",
    "LATOYA", "AALIYAH", "IMANI", "KEISHA", "JADA", "TAMIKA",
    "KWAME", "KOFI", "AMARA", "CHIDI", "NNAMDI", "FOLAKE", "ABEBE",
    # Additional Hispanic / Latino
    "MATEO", "SANTIAGO", "DIEGO", "JAVIER", "ALEJANDRO", "RICARDO", "FRANCISCO",
    "CAMILA", "VALENTINA", "GABRIELA", "LUCIA", "FERNANDA", "ROSA", "GUADALUPE",
]

_DIVERSE_LAST_NAMES = [
    # East Asian
    "WANG", "ZHANG", "LIU", "YANG", "HUANG", "ZHAO", "ZHOU", "SUN", "ZHU",
    "CHOI", "JEONG", "KANG", "YOON", "JANG", "HAN", "SHIN",
    "TRAN", "PHAM", "PHAN", "DANG", "BUI", "DUONG",
    # South Asian
    "SHARMA", "SINGH", "KUMAR", "GUPTA", "RAO", "REDDY", "NAIR", "IYER",
    "MEHTA", "SHAH", "DESAI", "JOSHI", "MALHOTRA", "BANERJEE", "DAS",
    # Arabic / Middle Eastern
    "KHAN", "RAHMAN", "ABDULLAH", "AZIZ", "SALEH", "NASSER", "HADDAD", "KHOURY",
    # African
    "OKAFOR", "OKONKWO", "ADEYEMI", "ADEBAYO", "MENSAH", "OSEI", "DIALLO",
    "TRAORE", "KEITA", "MWANGI", "OCHIENG", "GETACHEW",
    # Additional Hispanic / Latino
    "GUERRERO", "ROJAS", "NAVARRO", "CAMPOS", "VEGA", "CABRERA", "IBARRA",
    "CONTRERAS", "ESPINOZA", "JUAREZ", "AGUILAR", "PENA",
]


def _interleave(base, extra):
    """Spread ``extra`` names across the rank spectrum of ``base``.

    Name sampling uses Zipf weights keyed by list position, so simply appending
    the diverse names would relegate them all to the rare tail. Interleaving
    gives them a realistic mix of common and rare ranks. Deterministic.
    """
    if not extra:
        return list(base)
    ratio = max(1, len(base) // len(extra))
    out, ei = [], 0
    for i, name in enumerate(base):
        out.append(name)
        if (i + 1) % ratio == 0 and ei < len(extra):
            out.append(extra[ei])
            ei += 1
    out.extend(extra[ei:])
    return out


FIRST_NAMES = _interleave(FIRST_NAMES, _DIVERSE_FIRST_NAMES)
LAST_NAMES = _interleave(LAST_NAMES, _DIVERSE_LAST_NAMES)

REFERRAL_TYPES = ["GPS", "CPS"]
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Generic binary (one-hot) indicator fields. Semantics are irrelevant to the
# pipeline -- could be sex, race==X, "has a dog", etc. They flow through the
# existing exact-match feature path (features[0]) with no new comparison code.
DEFAULT_BINARY_FIELDS = ["bin1", "bin2"]


# ============================================================================
# Helper functions
# ============================================================================

def _zipf_weights(n, s=1.0):
    """Generate Zipf-like frequency weights for name sampling."""
    ranks = np.arange(1, n + 1, dtype=float)
    weights = 1.0 / (ranks ** s)
    return weights / weights.sum()


def _typo(name, rng):
    """Introduce a single-character typo into a name."""
    if len(name) < 2:
        return name
    action = rng.integers(0, 4)
    pos = rng.integers(0, len(name))
    if action == 0 and len(name) > 2:
        # drop a character
        return name[:pos] + name[pos + 1:]
    elif action == 1 and pos < len(name) - 1:
        # swap adjacent characters
        chars = list(name)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        return "".join(chars)
    elif action == 2:
        # substitute a character
        replacement = ALPHA[rng.integers(0, 26)]
        return name[:pos] + replacement + name[pos + 1:]
    else:
        # insert a character
        insertion = ALPHA[rng.integers(0, 26)]
        return name[:pos] + insertion + name[pos:]


def _random_date(rng, start_year=1960, end_year=2005):
    """Generate a random date as (year, month, day, datetime)."""
    year = int(rng.integers(start_year, end_year))
    month = int(rng.integers(1, 13))
    day = int(rng.integers(1, 29))  # avoid month-end edge cases
    dt = datetime(year, month, day)
    return year, month, day, dt


def _random_referral_date(rng, start_year=2018, end_year=2023):
    """Generate a random referral date."""
    year = int(rng.integers(start_year, end_year))
    month = int(rng.integers(1, 13))
    day = int(rng.integers(1, 29))
    return datetime(year, month, day)


def _make_base_records(n, rng, id_prefix="R", start_eid=0,
                       binary_fields=DEFAULT_BINARY_FIELDS, binary_rates=None):
    """
    Generate n unique base entity records as a list of dicts.

    Each record carries a unique ``true_entity_id`` (start_eid + i) -- the ground
    truth used to score precision/recall and to assert CPU/GPU output parity.
    """
    fn_weights = _zipf_weights(len(FIRST_NAMES), s=0.8)
    ln_weights = _zipf_weights(len(LAST_NAMES), s=0.7)

    fn_indices = rng.choice(len(FIRST_NAMES), size=n, p=fn_weights)
    ln_indices = rng.choice(len(LAST_NAMES), size=n, p=ln_weights)
    ssns = rng.integers(100_000_000, 999_999_999, size=n, endpoint=True)
    mciids = np.arange(1001, 1001 + n)
    counties = rng.integers(1, 68, size=n).astype(str)
    has_suffix = rng.random(n) < 0.05
    suffixes = np.where(has_suffix, rng.integers(1, 4, size=n).astype(float), np.nan)
    ref_types = rng.choice(REFERRAL_TYPES, size=n)

    if binary_rates is None:
        binary_rates = {f: 0.5 for f in binary_fields}
    bin_values = {f: (rng.random(n) < binary_rates[f]).astype(int)
                  for f in binary_fields}

    records = []
    for i in range(n):
        dobyy, dobmm, dobdd, dob = _random_date(rng)
        dt_ref = _random_referral_date(rng)
        rec = {
            "firstname": FIRST_NAMES[fn_indices[i]],
            "lastname": LAST_NAMES[ln_indices[i]],
            "ssn": float(ssns[i]),
            "mciid": float(mciids[i]),
            "county": counties[i],
            "dobyy": float(dobyy),
            "dobmm": float(dobmm),
            "dobdd": float(dobdd),
            "suffix": suffixes[i],
            "referralid": f"{id_prefix}{i:06d}",
            "referraltype": ref_types[i],
            "dob": dob,
            "dt_referral": dt_ref,
            "true_entity_id": start_eid + i,
        }
        for f in binary_fields:
            rec[f] = int(bin_values[f][i])
        records.append(rec)
    return records


def _vary_record(record, variation, rng,
                 binary_fields=DEFAULT_BINARY_FIELDS, new_eid=None):
    """
    Create a variant of a record based on the variation type.

    A variant inherits the source's ``true_entity_id`` (it is the same person)
    EXCEPT for ``sibling``, which is a different person and is assigned ``new_eid``.

    Variations are grouped by whether they survive blocking. The dedupe/match
    indexers block on EXACT firstname, lastname, or ssn (plus transposed
    firstname/lastname). A variant only becomes a candidate pair if at least one
    of those keys still matches exactly:

      survive blocking: exact, typo_fn, typo_ln, transposed, missing_ssn,
                        missing_mciid, diff_county, flip_binary, sibling,
                        nickname, phonetic_fn, phonetic_ln, ocr_fn, ocr_ln
                        (each alters at most one name field, so >=1 blocking key
                        stays exact)
      defeat blocking:  defeats_blocking (corrupts every blocking key -> the true
                        match is never even scored, i.e. an invisible false negative)
    """
    rec = record.copy()
    rec["referralid"] = f"V{rng.integers(100000, 999999)}"
    rec["dt_referral"] = _random_referral_date(rng)

    if variation == "exact":
        pass  # only referralid/dt_referral differ
    elif variation == "typo_fn":
        rec["firstname"] = _typo(rec["firstname"], rng)
    elif variation == "typo_ln":
        rec["lastname"] = _typo(rec["lastname"], rng)
    elif variation == "transposed":
        rec["firstname"], rec["lastname"] = rec["lastname"], rec["firstname"]
    elif variation == "missing_ssn":
        rec["ssn"] = np.nan
    elif variation == "missing_mciid":
        rec["mciid"] = np.nan
    elif variation == "diff_county":
        rec["county"] = str(rng.integers(1, 68))
    elif variation == "nickname":
        # Semantic first-name variant (ROBERT->BOB); falls below JW 0.9 but the
        # lastname/ssn block still catches it. Falls back to a typo on a miss.
        v = nickname_variant(rec["firstname"], rng)
        rec["firstname"] = v if v is not None else _typo(rec["firstname"], rng)
    elif variation == "phonetic_fn":
        v = phonetic_variant(rec["firstname"], rng, kind="first")
        rec["firstname"] = v if v is not None else _typo(rec["firstname"], rng)
    elif variation == "phonetic_ln":
        v = phonetic_variant(rec["lastname"], rng, kind="last")
        rec["lastname"] = v if v is not None else _typo(rec["lastname"], rng)
    elif variation == "ocr_fn":
        v = ocr_variant(rec["firstname"], rng)
        rec["firstname"] = v if v is not None else _typo(rec["firstname"], rng)
    elif variation == "ocr_ln":
        v = ocr_variant(rec["lastname"], rng)
        rec["lastname"] = v if v is not None else _typo(rec["lastname"], rng)
    elif variation == "flip_binary":
        # Clerical error: same person/name, one binary indicator recorded wrong.
        f = binary_fields[rng.integers(0, len(binary_fields))]
        rec[f] = 1 - int(rec[f])
    elif variation == "defeats_blocking":
        # Corrupt every blocking key so the pair is never generated. Same person
        # (entity id inherited) -> a true match the pipeline can never recover.
        rec["firstname"] = _typo(rec["firstname"], rng)
        rec["lastname"] = _typo(rec["lastname"], rng)
        rec["ssn"] = np.nan
    elif variation == "sibling":
        # Different person: same referralid (household), same lastname, but
        # different firstname/DOB/ssn. Gets its OWN entity id.
        fn_weights = _zipf_weights(len(FIRST_NAMES), s=0.8)
        rec["firstname"] = FIRST_NAMES[rng.choice(len(FIRST_NAMES), p=fn_weights)]
        rec["referralid"] = record["referralid"]  # keep same referralid
        dobyy, dobmm, dobdd, dob = _random_date(rng)
        rec["dobyy"], rec["dobmm"], rec["dobdd"], rec["dob"] = dobyy, dobmm, dobdd, dob
        rec["ssn"] = float(rng.integers(100_000_000, 999_999_999))
        rec["mciid"] = float(rng.integers(50000, 99999))
        if new_eid is not None:
            rec["true_entity_id"] = new_eid
    return rec


# ============================================================================
# Public generators
# ============================================================================

def generate_dedupe_data(n_records=10000, dedupe_rate=0.30, defeat_rate=0.05,
                         seed=42, binary_fields=DEFAULT_BINARY_FIELDS,
                         binary_rates=None):
    """
    Generate synthetic deduplication data.

    Parameters
    ----------
    n_records : int
        Total number of records to generate.
    dedupe_rate : float
        Fraction of records that are duplicates of existing entities.
    defeat_rate : float
        Fraction of duplicates whose every blocking key is corrupted so the true
        match never becomes a candidate pair (invisible false negatives). Lets you
        measure true recall rather than recall-among-blocked-pairs.
    seed : int
        Random seed for reproducibility.
    binary_fields : list of str
        Names of the generic binary (one-hot) indicator columns to emit.
    binary_rates : dict or None
        Optional {field: base_rate} for each binary field (default 0.5 each).

    Returns
    -------
    pd.DataFrame
        DataFrame ready for the crosswalk pipeline, including a ``true_entity_id``
        ground-truth column and the requested binary indicator columns.
    """
    rng = np.random.default_rng(seed)
    n_entities = int(n_records * (1 - dedupe_rate))
    n_dupes = n_records - n_entities

    # Generate unique base entities (entity ids 0 .. n_entities-1)
    base_records = _make_base_records(n_entities, rng, id_prefix="D", start_eid=0,
                                      binary_fields=binary_fields,
                                      binary_rates=binary_rates)

    # Survivable-variation distribution (each keeps >=1 blocking key exact).
    surv_types = ["exact", "typo_fn", "typo_ln", "transposed", "missing_ssn",
                  "diff_county", "flip_binary", "sibling", "nickname",
                  "phonetic_fn", "phonetic_ln", "ocr_fn", "ocr_ln"]
    surv_probs = [0.25, 0.15, 0.10, 0.07, 0.04, 0.04, 0.06, 0.04, 0.08,
                  0.05, 0.04, 0.04, 0.04]

    dupe_records = []
    source_indices = rng.integers(0, n_entities, size=n_dupes)
    defeats = rng.random(n_dupes) < defeat_rate
    surv_variations = rng.choice(surv_types, size=n_dupes, p=surv_probs)
    next_eid = n_entities  # siblings are new people -> fresh entity ids

    for i in range(n_dupes):
        source = base_records[source_indices[i]]
        if defeats[i]:
            variant = _vary_record(source, "defeats_blocking", rng, binary_fields)
        elif surv_variations[i] == "sibling":
            variant = _vary_record(source, "sibling", rng, binary_fields,
                                   new_eid=next_eid)
            next_eid += 1
        else:
            variant = _vary_record(source, surv_variations[i], rng, binary_fields)
        dupe_records.append(variant)

    all_records = base_records + dupe_records
    df = pd.DataFrame(all_records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["dob"] = pd.to_datetime(df["dob"])
    df["dt_referral"] = pd.to_datetime(df["dt_referral"])
    for col in ["dobyy", "dobmm", "dobdd"]:
        df[col] = df[col].astype(float)
    return df


def generate_match_data(n_records=10000, match_rate=0.85, defeat_rate=0.05,
                        seed=42, binary_fields=DEFAULT_BINARY_FIELDS,
                        binary_rates=None):
    """
    Generate synthetic matching data (two DataFrames).

    Parameters
    ----------
    n_records : int
        Number of records in each DataFrame.
    match_rate : float
        Fraction of dfB records that match a record in dfA.
    defeat_rate : float
        Fraction of matched dfB records whose every blocking key is corrupted so
        the true match never becomes a candidate pair (invisible false negatives).
    seed : int
        Random seed for reproducibility.
    binary_fields : list of str
        Names of the generic binary (one-hot) indicator columns to emit.
    binary_rates : dict or None
        Optional {field: base_rate} for each binary field (default 0.5 each).

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Tuple of (dfA, dfB). A matched dfB record shares its source's
        ``true_entity_id``; join on that column to recover ground-truth matches.
    """
    rng = np.random.default_rng(seed + 1000)  # different seed from dedupe
    n_matches = int(n_records * match_rate)
    n_unique_b = n_records - n_matches

    # dfA: all unique entities (entity ids 0 .. n_records-1)
    records_a = _make_base_records(n_records, rng, id_prefix="A", start_eid=0,
                                   binary_fields=binary_fields,
                                   binary_rates=binary_rates)

    # Survivable-variation distribution (each keeps >=1 blocking key exact).
    surv_types = ["exact", "typo_fn", "typo_ln", "transposed", "missing_ssn",
                  "missing_mciid", "diff_county", "flip_binary", "nickname",
                  "phonetic_fn", "phonetic_ln", "ocr_fn", "ocr_ln"]
    surv_probs = [0.25, 0.15, 0.10, 0.08, 0.04, 0.04, 0.04, 0.06, 0.08,
                  0.05, 0.05, 0.03, 0.03]

    # Pick which dfA records get matched
    match_indices = rng.choice(n_records, size=n_matches, replace=False)
    defeats = rng.random(n_matches) < defeat_rate
    surv_variations = rng.choice(surv_types, size=n_matches, p=surv_probs)

    matched_records = []
    for i in range(n_matches):
        source = records_a[match_indices[i]]
        if defeats[i]:
            variant = _vary_record(source, "defeats_blocking", rng, binary_fields)
        else:
            variant = _vary_record(source, surv_variations[i], rng, binary_fields)
        variant["referralid"] = f"B{i:06d}"
        matched_records.append(variant)

    # Unique-to-B records get fresh entity ids disjoint from dfA's
    unique_b_records = _make_base_records(n_unique_b, rng, id_prefix="U",
                                          start_eid=n_records,
                                          binary_fields=binary_fields,
                                          binary_rates=binary_rates)

    records_b = matched_records + unique_b_records

    dfA = pd.DataFrame(records_a)
    dfB = pd.DataFrame(records_b)
    dfB = dfB.sample(frac=1, random_state=seed).reset_index(drop=True)

    for df in [dfA, dfB]:
        df["dob"] = pd.to_datetime(df["dob"])
        df["dt_referral"] = pd.to_datetime(df["dt_referral"])
        for col in ["dobyy", "dobmm", "dobdd"]:
            df[col] = df[col].astype(float)

    return dfA, dfB
