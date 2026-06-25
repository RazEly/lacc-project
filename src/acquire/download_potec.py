"""Download the PoTeC eye-tracking recordings from OSF.

Assumes the PoTeC GitHub repo has already been cloned into ``data/potec`` (done
by ``init.sh``); the clone alone provides the stimuli / ``word_features`` TSVs
that ``domain_preprocessing`` needs. This script adds the large eye-tracking
data on top.

    python -m src.acquire.download_potec
"""
import sys

from src.config import POTEC_DIR, POTEC_EYETRACKING_DIR
from src.acquire.download_data_files import download_data


def download_eyetracking() -> None:
    print(f"Downloading PoTeC eye-tracking data -> {POTEC_EYETRACKING_DIR} ...")
    # Defaults mirror the upstream script: merged reading measures + scanpaths,
    # fixations and raw data.
    download_data(
        extract=True,
        output_folder=str(POTEC_EYETRACKING_DIR),
        download_asc=False,
        download_fixations=True,
        download_fixations_uncorrected=False,
        download_raw_data=True,
        download_reading_measures=False,
        download_reading_measures_merged=True,
        download_scanpaths=False,
        download_scanpaths_merged=True,
    )


def main() -> None:
    if not POTEC_DIR.exists():
        print(f"PoTeC checkout not found at {POTEC_DIR}. Clone it first "
              f"(init.sh does this).", file=sys.stderr)
        sys.exit(1)
    download_eyetracking()
    print("PoTeC eye-tracking data ready.")


if __name__ == "__main__":
    main()
