"""
Curated name-variant tables for realistic synthetic record-linkage data.

These tables capture variant types that character-level typos CANNOT produce:
  * nickname <-> formal name   (ROBERT -> BOB, MARGARET -> PEGGY)
  * phonetic / spelling variants (SARAH <-> SARA, CATHERINE <-> KATHRYN)
  * OCR / scanning confusions    (RN <-> M, CL <-> D)

The tables are generated once, offline (curated from general knowledge -- no LLM
or external service at runtime), and sampled DETERMINISTICALLY by the generator
via a seeded numpy Generator. This keeps generation reproducible and fast while
adding the semantic realism that ``generate_data._typo`` cannot.

All entries are UPPERCASE to match the standardized name pools in
``generate_data``.
"""

import numpy as np


# ============================================================================
# Nicknames: formal name -> list of common informal forms.
# Lookup is made bidirectional at import time (see _build_nickname_lookup).
# ============================================================================

_NICKNAMES = {
    # men
    "ROBERT": ["ROB", "BOB", "BOBBY", "BERT", "ROBBIE"],
    "WILLIAM": ["BILL", "WILL", "BILLY", "WILLIE", "LIAM"],
    "RICHARD": ["RICK", "DICK", "RICH", "RICHIE", "RICKY"],
    "JAMES": ["JIM", "JIMMY", "JAMIE", "JIMMIE"],
    "JOHN": ["JACK", "JOHNNY", "JOHNNIE"],
    "JOSEPH": ["JOE", "JOEY"],
    "THOMAS": ["TOM", "TOMMY"],
    "CHARLES": ["CHARLIE", "CHUCK", "CHAS"],
    "CHRISTOPHER": ["CHRIS", "TOPHER"],
    "DANIEL": ["DAN", "DANNY"],
    "MATTHEW": ["MATT", "MATTY"],
    "ANTHONY": ["TONY"],
    "DONALD": ["DON", "DONNIE"],
    "KENNETH": ["KEN", "KENNY"],
    "EDWARD": ["ED", "EDDIE", "TED", "NED"],
    "RONALD": ["RON", "RONNIE"],
    "JEFFREY": ["JEFF"],
    "JACOB": ["JAKE"],
    "NICHOLAS": ["NICK", "NICKY"],
    "STEPHEN": ["STEVE", "STEVIE"],
    "STEVEN": ["STEVE", "STEVIE"],
    "BENJAMIN": ["BEN", "BENNY", "BENJI"],
    "SAMUEL": ["SAM", "SAMMY"],
    "RAYMOND": ["RAY"],
    "GREGORY": ["GREG"],
    "ALEXANDER": ["ALEX", "AL", "XANDER", "SANDY"],
    "PATRICK": ["PAT", "PADDY"],
    "ANDREW": ["ANDY", "DREW"],
    "TIMOTHY": ["TIM", "TIMMY"],
    "GERALD": ["JERRY", "GERRY"],
    "GABRIEL": ["GABE"],
    "EUGENE": ["GENE"],
    "PHILIP": ["PHIL"],
    "RUSSELL": ["RUSS"],
    "VINCENT": ["VINCE", "VINNY"],
    "LOUIS": ["LOU", "LOUIE"],
    "ALBERT": ["AL", "BERT", "ALBIE"],
    "LAWRENCE": ["LARRY"],
    "ZACHARY": ["ZACH", "ZAC"],
    "DOUGLAS": ["DOUG"],
    "JONATHAN": ["JON", "JONNY"],
    "DENNIS": ["DENNY"],
    "HAROLD": ["HAL", "HARRY"],
    "HENRY": ["HANK", "HARRY"],
    "ARTHUR": ["ART", "ARTIE"],
    "PETER": ["PETE"],
    "FREDERICK": ["FRED", "FREDDIE"],
    "HOWARD": ["HOWIE"],
    "MARTIN": ["MARTY"],
    "FRANCIS": ["FRANK", "FRANKIE"],
    "LEONARD": ["LEN", "LENNY"],
    # women
    "MARY": ["MOLLY", "POLLY"],
    "PATRICIA": ["PAT", "PATTY", "TRICIA", "TRISH"],
    "JENNIFER": ["JEN", "JENNY", "JENNIE"],
    "ELIZABETH": ["LIZ", "BETH", "BETTY", "LIZZIE", "ELIZA", "LIBBY"],
    "BARBARA": ["BARB", "BARBIE"],
    "SUSAN": ["SUE", "SUSIE", "SUZY"],
    "JESSICA": ["JESS", "JESSIE"],
    "MARGARET": ["MEG", "MAGGIE", "PEGGY", "MARGE", "GRETA"],
    "SANDRA": ["SANDY"],
    "ASHLEY": ["ASH"],
    "KIMBERLY": ["KIM"],
    "EMILY": ["EM", "EMMY"],
    "MICHELLE": ["SHELLY", "MICKY"],
    "DOROTHY": ["DOT", "DOTTIE", "DORA"],
    "AMANDA": ["MANDY"],
    "MELISSA": ["MEL", "MISSY", "LISSA"],
    "DEBORAH": ["DEB", "DEBBIE", "DEBRA"],
    "STEPHANIE": ["STEPH", "STEFFIE"],
    "REBECCA": ["BECKY", "BECCA", "REBA"],
    "CYNTHIA": ["CINDY"],
    "KATHLEEN": ["KATHY", "KATE", "KAT"],
    "ANGELA": ["ANGIE"],
    "ANNA": ["ANNIE", "ANN"],
    "PAMELA": ["PAM"],
    "NICOLE": ["NIKKI", "NICKY"],
    "SAMANTHA": ["SAM", "SAMMY"],
    "CHRISTINE": ["CHRIS", "CHRISSY", "TINA"],
    "CHRISTINA": ["CHRIS", "TINA", "CHRISSY"],
    "CATHERINE": ["CATHY", "KATE", "CATE", "KIT", "KATIE"],
    "KATHERINE": ["KATHY", "KATE", "KATIE", "KIT"],
    "KATHRYN": ["KATE", "KATIE", "KATHY"],
    "CAROLYN": ["CAROL", "LYNN"],
    "JANET": ["JAN"],
    "DIANE": ["DI"],
    "RUTH": ["RUTHIE"],
    "VIRGINIA": ["GINNY", "GINGER"],
    "VICTORIA": ["VICKY", "TORI", "VICKIE"],
    "JUDITH": ["JUDY"],
    "MEGAN": ["MEG"],
    "JACQUELINE": ["JACKIE", "JACQUI"],
    "MARTHA": ["MARTY", "MATTIE"],
    "TERESA": ["TERRY", "TESS"],
    "THERESA": ["TERRY", "TESS"],
    "ABIGAIL": ["ABBY", "ABBIE", "GAIL"],
    "ISABELLA": ["BELLA", "IZZY", "ISA"],
    "DANIELLE": ["DANI"],
    "NATALIE": ["NAT", "NATTY"],
    "CHARLOTTE": ["CHARLIE", "LOTTIE", "CHAR"],
    "ALEXIS": ["LEXI", "ALEX"],
    "SOPHIA": ["SOPHIE"],
    "GRACE": ["GRACIE"],
    "OLIVIA": ["LIV", "OLLIE", "LIVVY"],
    "MADISON": ["MADDIE"],
    "BRITTANY": ["BRITT"],
    "EVELYN": ["EVE", "EVIE"],
    "FRANCES": ["FRAN", "FRANNY"],
    "JANICE": ["JAN"],
    "ANDREA": ["ANDY", "DREA"],
    "BEVERLY": ["BEV"],
    "JOAN": ["JOANIE"],
}


def _build_nickname_lookup(table):
    """Build a bidirectional name -> [alternatives] lookup from the formal map."""
    lookup = {}
    for formal, nicks in table.items():
        lookup.setdefault(formal, set()).update(nicks)
        for n in nicks:
            lookup.setdefault(n, set()).add(formal)
    return {k: sorted(v) for k, v in lookup.items()}


_NICK_LOOKUP = _build_nickname_lookup(_NICKNAMES)


# ============================================================================
# Phonetic / spelling-variant equivalence groups. Any member can be swapped for
# any other in the same group. Kept separate for first vs last names.
# ============================================================================

_PHONETIC_FIRST_GROUPS = [
    ["SARAH", "SARA"],
    ["CATHERINE", "KATHERINE", "KATHRYN", "CATHRYN", "KATHARINE"],
    ["SEAN", "SHAWN", "SHAUN"],
    ["ERIC", "ERIK", "ERICK"],
    ["MARK", "MARC"],
    ["STEPHEN", "STEVEN"],
    ["PHILIP", "PHILLIP"],
    ["ELIZABETH", "ELISABETH"],
    ["HANNAH", "HANNA"],
    ["AARON", "ARON", "ARRON"],
    ["BRIAN", "BRYAN"],
    ["MEGAN", "MEAGAN", "MEGHAN"],
    ["MICHELLE", "MICHELE"],
    ["ASHLEY", "ASHLEIGH"],
    ["JACOB", "JAKOB"],
    ["ISABELLA", "ISABELA"],
    ["ALAN", "ALLAN", "ALLEN"],
    ["DENISE", "DENICE"],
    ["RACHEL", "RACHAEL"],
    ["TERESA", "THERESA"],
    ["KIMBERLY", "KIMBERLEY"],
    ["JEFFREY", "GEOFFREY"],
    ["CAROLYN", "CAROLYNN"],
    ["NATALIE", "NATALEE"],
    ["GERALD", "GERALDO"],
    # broader-origin spelling variants
    ["MOHAMMED", "MUHAMMAD", "MOHAMED", "MOHAMMAD"],
    ["AISHA", "AYESHA", "AYSHA"],
    ["OMAR", "OMER"],
    ["FATIMA", "FATIMAH"],
    ["ALI", "ALY"],
    ["YUSUF", "YOUSEF", "YUSIF"],
    ["IBRAHIM", "IBRAHEEM"],
    ["LAYLA", "LEILA", "LAILA"],
    ["MATEO", "MATTEO"],
    ["RAHUL", "RAHOOL"],
    ["AALIYAH", "ALIYAH"],
]

_PHONETIC_LAST_GROUPS = [
    ["SMITH", "SMYTH", "SMITHE"],
    ["BROWN", "BROWNE"],
    ["CLARK", "CLARKE"],
    ["GREEN", "GREENE"],
    ["DAVIS", "DAVIES"],
    ["THOMPSON", "THOMSON"],
    ["JOHNSON", "JOHNSEN", "JONSON"],
    ["ANDERSON", "ANDERSEN"],
    ["CARLSON", "CARLSEN"],
    ["HANSEN", "HANSON"],
    ["JENSEN", "JENSON"],
    ["OLSON", "OLSEN", "OLSSON"],
    ["MEYER", "MYER", "MEYERS"],
    ["REED", "REID"],
    ["LEE", "LEIGH"],
    ["MARTIN", "MARTYN"],
    ["STEWART", "STUART"],
    ["PHILLIPS", "PHILIPS"],
    ["NICHOLS", "NICHOLLS", "NICOLS"],
    ["PETERSON", "PETERSEN"],
    ["WHITE", "WHYTE"],
    ["MORGAN", "MORGEN"],
]


def _build_group_lookup(groups):
    """Map each name to the other members of its equivalence group."""
    lookup = {}
    for group in groups:
        for name in group:
            lookup[name] = [g for g in group if g != name]
    return lookup


_PHON_FIRST = _build_group_lookup(_PHONETIC_FIRST_GROUPS)
_PHON_LAST = _build_group_lookup(_PHONETIC_LAST_GROUPS)


# ============================================================================
# OCR / scanning confusions: substring pairs commonly misread in scanned forms.
# Applied to a single randomly chosen matching position.
# ============================================================================

_OCR_PAIRS = [
    ("RN", "M"), ("M", "RN"),
    ("CL", "D"), ("D", "CL"),
    ("VV", "W"), ("W", "VV"),
    ("LI", "U"),
    ("I", "L"), ("L", "I"),
    ("O", "Q"), ("Q", "O"),
    ("E", "F"),
    ("B", "R"),
    ("U", "V"),
]


# ============================================================================
# Public sampling helpers. Each returns a NEW variant string, or None when the
# input has no entry in the relevant table (caller should fall back to a typo).
# ============================================================================

def nickname_variant(name, rng):
    """Return a nickname/formal alternative for ``name``, or None."""
    alts = _NICK_LOOKUP.get(name)
    if not alts:
        return None
    return alts[int(rng.integers(0, len(alts)))]


def phonetic_variant(name, rng, kind="first"):
    """Return a phonetic/spelling variant for ``name``, or None."""
    table = _PHON_FIRST if kind == "first" else _PHON_LAST
    alts = table.get(name)
    if not alts:
        return None
    return alts[int(rng.integers(0, len(alts)))]


def ocr_variant(name, rng):
    """Apply one OCR confusion to ``name`` at a random matching position, or None."""
    candidates = []
    for src, dst in _OCR_PAIRS:
        start = 0
        while True:
            idx = name.find(src, start)
            if idx == -1:
                break
            candidates.append((idx, src, dst))
            start = idx + 1
    if not candidates:
        return None
    idx, src, dst = candidates[int(rng.integers(0, len(candidates)))]
    return name[:idx] + dst + name[idx + len(src):]
