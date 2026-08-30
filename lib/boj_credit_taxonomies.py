from __future__ import annotations

from typing import Any


PRIMARY_TAXONOMY_ID = "bezemer_samarina_zhang_2020_japan_v1"
WERNER_TAXONOMY_ID = "werner_1997_financial_circulation_v1"
MULLER_VERNER_TAXONOMY_ID = "muller_verner_2024_sector_v1"
LEGACY_TAXONOMY_ID = "author_gbe_legacy_v1"
MAPPING_VERSION = "boj-bezemer-four-way-common-population-2026-07-30-v1"

OFFICIAL_TOTAL_STOCK_CODE = "DLLILKG90_DLLI5DS2T"
OFFICIAL_TOTAL_FIXED_CODE = "DLLILKG90_DLLI5DS5T"
HOUSEHOLD_TOTAL_STOCK_CODE = "DLLILKG62_DLLI5DS2TPN"
HOUSEHOLD_HOUSING_STOCK_CODE = "DLHLLKG71_DLHL2DSSL"
HOUSEHOLD_CONSUMER_STOCK_CODE = "DLCLLKG71_DLCL2DSTSL"

PRIMARY_BUCKETS = ("NFB", "FIN", "PROP", "HH_NONHOUSING")
WERNER_BUCKETS = ("FCP", "COMPLEMENT")
MULLER_VERNER_BUCKETS = (
    "TRADABLE",
    "NONTRADABLE",
    "OTHER_NFB",
    "FIN",
    "HH",
    "UNRESOLVED",
)


# The mappings below are fixed from published definitions and BOJ borrower-sector
# labels. No outcome, forecast, or OOS result enters this table.
SECTOR_MAP: tuple[dict[str, str], ...] = (
    {
        "bucket": "G",
        "sector": "manufacturing",
        "stock_code": "DLLILKG21_DLLI5DS2TMK",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "TRADABLE",
    },
    {
        "bucket": "G",
        "sector": "agriculture_forestry",
        "stock_code": "DLLILKG86_DLLI5DS2TAF",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "TRADABLE",
    },
    {
        "bucket": "G",
        "sector": "fishery",
        "stock_code": "DLLILKG24_DLLI5DS2TFS",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "TRADABLE",
    },
    {
        "bucket": "G",
        "sector": "mining_quarrying",
        "stock_code": "DLLILKG25_DLLI5DS2TMN",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "TRADABLE",
    },
    {
        "bucket": "G",
        "sector": "electricity_gas_water",
        "stock_code": "DLLILKG29_DLLI5DS2TEG",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "OTHER_NFB",
    },
    {
        "bucket": "G",
        "sector": "information_communications",
        "stock_code": "DLLILKG74_DLLI5DS2TIC",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "NONTRADABLE",
    },
    {
        "bucket": "G",
        "sector": "transport_postal",
        "stock_code": "DLLILKG75_DLLI5DS2TTR",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "NONTRADABLE",
    },
    {
        "bucket": "G",
        "sector": "wholesale_trade",
        "stock_code": "DLLILKG40_DLLI5DS2TWS",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "NONTRADABLE",
    },
    {
        "bucket": "G",
        "sector": "retail_trade",
        "stock_code": "DLLILKG43_DLLI5DS2TRT",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "NONTRADABLE",
    },
    {
        "bucket": "G",
        "sector": "goods_rental_leasing",
        "stock_code": "DLLILKG51_DLLI5DS2TLS",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "OTHER_NFB",
    },
    {
        "bucket": "G",
        "sector": "professional_technical_services",
        "stock_code": "DLLILKG97_DLLI5DS2TSC",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "OTHER_NFB",
    },
    {
        "bucket": "G",
        "sector": "hotels",
        "stock_code": "DLLILKG52_DLLI5DS2THL",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "NONTRADABLE",
    },
    {
        "bucket": "G",
        "sector": "eating_drinking_services",
        "stock_code": "DLLILKG44_DLLI5DS2TRS",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "NONTRADABLE",
    },
    {
        "bucket": "G",
        "sector": "living_personal_amusement_services",
        "stock_code": "DLLILKG98_DLLI5DS2TLA",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "OTHER_NFB",
    },
    {
        "bucket": "G",
        "sector": "education_learning_support",
        "stock_code": "DLLILKG78_DLLI5DS2TED",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "OTHER_NFB",
    },
    {
        "bucket": "G",
        "sector": "medical_health_welfare",
        "stock_code": "DLLILKG77_DLLI5DS2TME",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "OTHER_NFB",
    },
    {
        "bucket": "G",
        "sector": "other_services",
        "stock_code": "DLLILKG79_DLLI5DS2TMS",
        "primary_bucket": "NFB",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "OTHER_NFB",
    },
    {
        "bucket": "G",
        "sector": "other_organizations",
        "stock_code": "DLLILKG80_DLLI5DS2TRE",
        "primary_bucket": "NFB_RESIDUAL_COMPONENT",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "UNRESOLVED",
    },
    {
        "bucket": "G",
        "sector": "local_governments",
        "stock_code": "DLLILKG61_DLLI5DS2TRO",
        "primary_bucket": "EXCLUDE_LOCAL_GOVERNMENT",
        "werner_bucket": "EXCLUDE_LOCAL_GOVERNMENT",
        "muller_verner_bucket": "EXCLUDE_LOCAL_GOVERNMENT",
    },
    {
        "bucket": "B",
        "sector": "construction",
        "stock_code": "DLLILKG26_DLLI5DS2TCT",
        "primary_bucket": "NFB",
        "werner_bucket": "FCP",
        "muller_verner_bucket": "NONTRADABLE",
    },
    {
        "bucket": "E",
        "sector": "finance_insurance",
        "stock_code": "DLLILKG49_DLLI5DS2TFI",
        "primary_bucket": "FIN",
        "werner_bucket": "FCP",
        "muller_verner_bucket": "FIN",
    },
    {
        "bucket": "E",
        "sector": "real_estate",
        "stock_code": "DLLILKG50_DLLI5DS2TFX",
        "primary_bucket": "PROP",
        "werner_bucket": "FCP",
        "muller_verner_bucket": "NONTRADABLE",
    },
    {
        "bucket": "E",
        "sector": "households_housing_consumer_tax",
        "stock_code": HOUSEHOLD_TOTAL_STOCK_CODE,
        "primary_bucket": "HOUSEHOLD_SPLIT",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "HH",
    },
    {
        "bucket": "U",
        "sector": "overseas_yen_and_transferred_loans",
        "stock_code": "DLLILKG63_DLLI5DS2TFL",
        "primary_bucket": "NFB_RESIDUAL_COMPONENT",
        "werner_bucket": "COMPLEMENT",
        "muller_verner_bucket": "UNRESOLVED",
    },
)


TAXONOMY_METADATA: dict[str, dict[str, Any]] = {
    PRIMARY_TAXONOMY_ID: {
        "role": "primary",
        "citation": (
            "Bezemer, Samarina, and Zhang (2020), Journal of Banking & Finance "
            "113, 105760, https://doi.org/10.1016/j.jbankfin.2020.105760; "
            "Japan crosswalk documented in DNB Working Paper 559."
        ),
        "population": (
            "Total outstanding loans of domestically licensed banks less local-government "
            "loans, exactly exhausted by non-financial business, financial business, "
            "property/mortgage, and household non-housing. Domestically licensed describes "
            "the lenders, not borrower geography; the residual NFB series includes the "
            "explicit overseas-linked component."
        ),
        "buckets": {
            "NFB": (
                "Published Japan residual: official total less finance/insurance, "
                "real estate, local government, and total household loans. Construction "
                "is included. Mapped NFB industries, overseas-linked loans, and the "
                "remaining unresolved component are disclosed separately."
            ),
            "FIN": "Finance and insurance borrowers.",
            "PROP": "Real-estate borrowers plus purpose-coded household housing loans.",
            "HH_NONHOUSING": (
                "Total household borrowing less purpose-coded household housing loans; "
                "this is not labelled consumer credit."
            ),
        },
        "exclusions": {
            "local_governments": "Reported separately, outside the private-credit population.",
        },
        "embedded_nfb_components": {
            "overseas_yen_and_transferred_loans": (
                "Explicitly disclosed because the published residual formula absorbs this "
                "non-domestic borrower category in the current BOJ total."
            ),
            "unresolved_residual": (
                "Explicitly disclosed residual after mapped NFB industries and the "
                "overseas-linked series are removed from published residual NFB."
            ),
        },
        "construction_placement": (
            "Construction is an NFB detail in the primary crosswalk; it is not a "
            "stand-alone purpose bucket."
        ),
    },
    WERNER_TAXONOMY_ID: {
        "role": "literature_anchored_robustness",
        "citation": (
            "Werner (1997), Kredit und Kapital 30(2), 276-309, "
            "https://doi.org/10.3790/ccm.30.2.276."
        ),
        "population": f"Exactly the included population of {PRIMARY_TAXONOMY_ID}.",
        "buckets": {
            "FCP": (
                "Werner-inspired BOJ borrower-sector proxy: construction, real estate, "
                "and the available finance/insurance aggregate. Werner's original Japan "
                "measure referred to non-bank financial institutions, so this is not an "
                "exact series match."
            ),
            "COMPLEMENT": (
                "Complement within the same included population, including disclosed "
                "overseas-linked and unresolved residual components."
            ),
        },
        "construction_placement": (
            "Construction enters the Werner-inspired financial-circulation proxy; the "
            "available BOJ finance/insurance series is broader than Werner's original "
            "non-bank-financial-institution component."
        ),
    },
    MULLER_VERNER_TAXONOMY_ID: {
        "role": "literature_anchored_robustness",
        "citation": (
            "Muller and Verner (2024), Review of Economic Studies 91(6), "
            "https://doi.org/10.1093/restud/rdad112."
        ),
        "population": f"Exactly the included population of {PRIMARY_TAXONOMY_ID}.",
        "buckets": {
            "TRADABLE": "Agriculture, fishery, mining, and manufacturing NFB.",
            "NONTRADABLE": (
                "Construction, real estate, trade, hotels/restaurants, transport, "
                "and information/communications."
            ),
            "OTHER_NFB": "Remaining mapped NFB industries.",
            "FIN": "Finance/insurance retained to exhaust the common population.",
            "HH": "Total household borrowing retained to exhaust the common population.",
            "UNRESOLVED": (
                "Overseas-linked and unresolved components retained explicitly to exhaust "
                "the common population; outside the paper's core sector contrast."
            ),
        },
        "construction_placement": (
            "Construction is nontradable, following the published tradable/nontradable "
            "sector crosswalk."
        ),
    },
    LEGACY_TAXONOMY_ID: {
        "role": "appendix_legacy",
        "citation": "Author-defined G/B/E grouping retained only for archive compatibility.",
        "population": "Mapped legacy G/B/E borrower groups.",
        "buckets": {
            "G": "Broad sectors, local governments, and other organisations.",
            "B": "Construction.",
            "E": "Finance/insurance, real estate, and households.",
        },
        "construction_placement": "Construction is separate in this legacy grouping.",
    },
}


TAXONOMY_SELECTION_RULE: dict[str, Any] = {
    "primary_taxonomy_id": PRIMARY_TAXONOMY_ID,
    "robustness_taxonomy_ids": [WERNER_TAXONOMY_ID, MULLER_VERNER_TAXONOMY_ID],
    "legacy_taxonomy_id": LEGACY_TAXONOMY_ID,
    "selection_basis": (
        "literature-anchored mappings reported jointly; the current-sample declaration "
        "is not an externally time-stamped preregistration"
    ),
    "outcome_columns_used": [],
    "oos_results_used": False,
}


__all__ = [
    "HOUSEHOLD_CONSUMER_STOCK_CODE",
    "HOUSEHOLD_HOUSING_STOCK_CODE",
    "HOUSEHOLD_TOTAL_STOCK_CODE",
    "LEGACY_TAXONOMY_ID",
    "MAPPING_VERSION",
    "MULLER_VERNER_BUCKETS",
    "MULLER_VERNER_TAXONOMY_ID",
    "OFFICIAL_TOTAL_FIXED_CODE",
    "OFFICIAL_TOTAL_STOCK_CODE",
    "PRIMARY_BUCKETS",
    "PRIMARY_TAXONOMY_ID",
    "SECTOR_MAP",
    "TAXONOMY_METADATA",
    "TAXONOMY_SELECTION_RULE",
    "WERNER_BUCKETS",
    "WERNER_TAXONOMY_ID",
]
