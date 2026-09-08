"""Where the raw data comes from.

Both sources are public UK open data under the Open Government Licence v3.0.
Nothing here is personal, household-level or sensitive: this is a methods
rehearsal on aggregate published statistics.

1. MHCLG Statutory Homelessness (H-CLIC) detailed local authority tables,
   annual financial-year releases.
   https://www.gov.uk/government/statistical-data-sets/live-tables-on-homelessness
2. MHCLG English Indices of Deprivation 2019 (IoD2019).
   https://www.gov.uk/government/statistics/english-indices-of-deprivation-2019

URLs are the media links gov.uk serves the files from, kept verbatim so they
can be checked against the publication page. Files are content-addressed by
gov.uk, so a URL changing means the publication was revised - which is a fact
worth noticing rather than papering over.
"""
from __future__ import annotations

BASE = "https://assets.publishing.service.gov.uk/media"

# --- H-CLIC detailed LA tables, one per financial year ------------------------
HCLIC = {
    "2018-19": f"{BASE}/5f749886e90e0740d7ebee40/DetailedLA_2018-2019.ods",
    "2019-20": f"{BASE}/61b0e5208fa8f5037b09c769/DetailedLA_2019-20__revised__Revised_dropdowns_fixed.ods",
    "2020-21": f"{BASE}/632c80778fa8f51d1ddaf81e/Detailed_LA_2020-21__revised_.ods",
    "2021-22": f"{BASE}/6549195dbdb7ef00124af914/Detailed_LA_2021-22__Revised_Nov_2023_.ods",
    "2022-23": f"{BASE}/6a2c1f981f6fa5c3377e5eb4/Detailed_LA_20222023_revised_corrected.ods",
    "2023-24": f"{BASE}/6a2c1f6aa3674dfd3eb5081f/Statutory_Homelessness_Detailed_Local_Authority_Data_2023-2024_Revised.ods",
    "2024-25": f"{BASE}/6a2ff87cd95ffddb05d4b063/Statutory_Homelessness_Detailed_Local_Authority_Data_2024-2025_corrected.ods",
}

# --- Indices of Deprivation 2019 ---------------------------------------------
IOD = {
    # LSOA-level: every score, rank and decile plus population denominators
    "iod2019_lsoa_all.csv":
        f"{BASE}/5dc407b440f0b6379a7acc8d/File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators_3.csv",
    # Lower-tier local authority district summaries
    "iod2019_lad_summaries.xlsx":
        f"{BASE}/5d8b3cfbe5274a08be69aa91/File_10_-_IoD2019_Local_Authority_District_Summaries__lower-tier__.xlsx",
}
