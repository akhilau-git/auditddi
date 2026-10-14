"""Collect DrugBank API data for all drugs in DDInter and save to Google Drive."""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
from requests.exceptions import RequestException

# Import our API client and path resolver
from drugbank_api import get_discovery_interactions
from path_resolver import resolve_data_base, resolve_results_base

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_unique_drugbank_ids(ddinter_csv_path: Path) -> list[str]:
    """Extract all unique DrugBank IDs from the DDInter normalized CSV."""
    if not ddinter_csv_path.exists():
        raise FileNotFoundError(f"Cannot find DDInter CSV at {ddinter_csv_path}")
    
    df = pd.read_csv(ddinter_csv_path, dtype=str)
    
    # DDInter normalized CSV has drug1_id and drug2_id
    ids_1 = df['drug1_id'].dropna().unique()
    ids_2 = df['drug2_id'].dropna().unique()
    
    # Combine and deduplicate
    all_ids = set(ids_1).union(set(ids_2))
    
    # Filter out anything that doesn't look like a DrugBank ID (DB...)
    drugbank_ids = [db_id for db_id in all_ids if db_id.startswith("DB")]
    
    return sorted(drugbank_ids)


def collect_drugbank_data(input_csv: Path, output_dir: Path, api_key: str | None = None) -> None:
    """Fetch and save DrugBank interactions for all IDs in the input CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    drugbank_ids = get_unique_drugbank_ids(input_csv)
    logging.info(f"Found {len(drugbank_ids)} unique DrugBank IDs in {input_csv.name}")
    
    success_count = 0
    error_count = 0
    
    for db_id in drugbank_ids:
        output_file = output_dir / f"{db_id}_interactions.json"
        
        # Skip if we already downloaded this drug's data (resumable)
        if output_file.exists():
            logging.debug(f"Skipping {db_id}, already downloaded.")
            success_count += 1
            continue
            
        logging.info(f"Fetching interactions for {db_id}...")
        try:
            # Query the Discovery API
            response = get_discovery_interactions(db_id, api_key=api_key)
            
            # Save the JSON response
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(response.json(), f, indent=2)
                
            success_count += 1
            
            # Be nice to the API (avoid rate limits)
            time.sleep(1.0)
            
        except RequestException as e:
            logging.error(f"Failed to fetch data for {db_id}: {e}")
            error_count += 1
            # Wait a bit longer if we hit an error
            time.sleep(5.0)
            
    logging.info(f"Collection complete. Success: {success_count}, Errors: {error_count}")
    logging.info(f"Data saved to Google Drive mapped folder: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DrugBank Discovery API data for DDInter drugs.")
    parser.add_argument("--api-key", help="DrugBank API Key (or set AUDITDDI_DRUGBANK_API_KEY env var)")
    args = parser.parse_args()
    
    # 1. Use the universal path resolver to find where DDInter data is
    # It will automatically find the local folder or Colab Google Drive path
    try:
        data_base = resolve_data_base()
        ddinter_input_csv = data_base / "DDInter" / "ddinter_interactions_normalized.csv"
    except FileNotFoundError:
        # Fallback for testing if data_base isn't perfectly mapped
        ddinter_input_csv = Path("../../auditddi-data/DDInter/ddinter_interactions_normalized.csv").resolve()

    # 2. Use path resolver to target the results folder (synced to Google Drive)
    results_base = resolve_results_base()
    drugbank_output_dir = results_base / "DrugBank" / "discovery_api_raw"
    
    logging.info(f"Input DDInter file: {ddinter_input_csv}")
    logging.info(f"Output DrugBank folder (Google Drive): {drugbank_output_dir}")
    
    collect_drugbank_data(ddinter_input_csv, drugbank_output_dir, args.api_key)


if __name__ == "__main__":
    main()
