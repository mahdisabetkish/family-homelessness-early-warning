"""Every path, year and threshold the pipeline uses, defined once.

Before this module existed the financial years were listed in three stages,
the train/validation/test split in two, and the Colchester authority code in
three. Each copy was correct, which is the dangerous case: a split year moved
in one place and left everywhere else would not raise, it would quietly
publish a table describing a split that was no longer the one being run.

Nothing here is computed from data. Anything that depends on what the files
actually contain belongs in the module that reads them.
"""
from __future__ import annotations

from pathlib import Path

# --- layout -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
DOCS = ROOT / "docs"
DASHBOARD_DATA = DOCS / "data"

HCLIC_RAW = RAW / "hclic"
IOD_RAW = RAW / "iod2019"

PANEL_FILE = PROCESSED / "panel.parquet"
DESIGN_FILE = PROCESSED / "design.parquet"
COLCHESTER_LSOA_FILE = PROCESSED / "colchester_lsoa.parquet"
ESSEX_LSOA_FILE = PROCESSED / "essex_lsoa.parquet"

# --- time -------------------------------------------------------------------
# H-CLIC financial years, oldest first. 2018-19 is the first year the Homelessness
# Reduction Act duties were in force, so it is the first year the measures mean
# what they mean now.
YEARS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]

# Published tables read from each release.
TABLES_READ = ["A1", "A2R", "A3", "A4R", "A5P", "A5R"]

# --- the modelling problem --------------------------------------------------
# Families with dependent children newly owed a relief duty - that is, already
# homeless - per 1,000 resident households.
OUTCOME = "rate_relief_with_children"

# The split is by the year the *outcome* falls in, not the year the features
# come from, so a model never sees a year at or after the one it is scored on.
# Outcome year t is predicted from features in year t-1, so training on
# outcomes 2020-21 onwards means features from 2019-20 onwards.
TRAIN_OUTCOME_YEARS = ["2020-21", "2021-22", "2022-23"]
VALID_OUTCOME_YEAR = "2023-24"
TEST_OUTCOME_YEAR = "2024-25"

ESCALATION_THRESHOLD = 1.25   # a 25% or greater relative rise
LEVEL_QUINTILE = 0.80         # worst 20% nationally
CAPACITY = 30                 # authorities a national team could actually work with

BOOTSTRAP_DRAWS = 2000
RANDOM_SEED = 0

# The model the dashboard quotes when they quote one.
HEADLINE_MODEL = "Gradient boosting (calibrated)"

# --- geography --------------------------------------------------------------
COLCHESTER = "E07000071"

# Essex county districts, used as the comparison group for the neighbourhood
# layer. Codes are ONS local authority district codes on 2019 boundaries.
ESSEX = {
    "E07000066": "Basildon", "E07000067": "Braintree", "E07000068": "Brentwood",
    "E07000069": "Castle Point", "E07000070": "Chelmsford", "E07000071": "Colchester",
    "E07000072": "Epping Forest", "E07000073": "Harlow", "E07000074": "Maldon",
    "E07000075": "Rochford", "E07000076": "Tendring", "E07000077": "Uttlesford",
}
