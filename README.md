# crosswalk
Authored Hyun Woo Kim, The Pennsylvania State University, 2018-2019
A library that performs probabilistic record linkage across non relational datasets. Built for linkages between PA administrative data, but is generalizable to any data set containing first and last names and other idenitifiers.
_____

Code review Alex Winters, The Pennsylvania State University, 2020-2021  
## Updates and Enhancements

### 1. `sarahs_rule` Function (`preprocessing.py`)
This function was updated to identify and document biological mothers based on specific rules and merge results into the `REL` DataFrame.

1. **Data Preparation**  
   - Merges `minors` and `majors` DataFrames on `referralid` and filters relevant rows.

2. **Rule-Based Identification**  
   - **Rule 1**: Identifies mothers with specific roles and relationships.  
   - **Rule 2**: Finds mothers based on age difference and removes duplicates.  
   - **Rule 3**: Matches mothers using surname consistency.

3. **Final Steps**  
   - Combines results, analyzes overlaps with existing data, and appends new relationships to the `REL` DataFrame.

---

### 2. `standardize` Function (`preprocessing.py`)
Refactored to ensure consistent name formatting and validation:

1. Converts names to uppercase and trims whitespace.  
2. Handles exceptions for known patterns and weak first names.  
3. Removes records with agency-related terms or invalid names.  
4. Standardizes numeric fields (`ssn`, `suffix`, `mciid`) and validates date consistency.

---

### 3. `population_split` Function (`preprocessing.py`)
Splits the population into minors and adults using combined age and role conditions:

1. **Temporary Columns**  
   - `_minor1` and `_minor2` determine underage status.

2. **Data Splitting**  
   - Splits data into `minors` and `majors`.

3. **Validation**  
   - Ensures the split is consistent by checking the shape of the resulting DataFrames.

---

### 4. `children` Function (`preprocessing.py`)
Improved to deduplicate and prioritize parent-child relationships:

1. **Parent Identification**  
   - Applies priority rules to identify the best mother and father for each child.

2. **Name Matching**  
   - Uses Soundex to match names for further deduplication.

3. **Data Merging**  
   - Merges additional family information for a comprehensive output.

---

### 5. `prep` Function (`evaluation.py`)
Enhanced for improved parameter handling and data cleaning:

1. Validates and merges optional variables.  
2. Removes NaN values and ensures numeric conversion.  
3. Optimized to drop self-duplicates and redundant rows.

---

### 6. `precision_recall` and `clerical_review` Functions (`evaluation.py`)
1. **`precision_recall`**  
   - Added return values for accuracy, precision, recall, and F1 score.

2. **`clerical_review`**  
   - Improved handling of false negatives and concatenated results accurately.

---

### 7. General Updates (`evaluation.py`)
1. Standardized import statements (`pandas as pd`, `numpy as np`).  
2. Minor bug fixes and performance improvements in multiple functions (`pr_tradeoff`, `clerical_review`).  
3. Added comments for better code clarity and readability.
