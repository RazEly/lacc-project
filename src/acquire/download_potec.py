"""Download the PoTeC eye-tracking recordings from OSF.

Assumes the PoTeC repo is already cloned into ``data/potec`` (init.sh); that
provides the stimuli / ``word_features`` TSVs. This adds the eye-tracking data.

    python -m src.acquire.download_potec
"""
import importlib.util
import sys

from src.config import POTEC_DIR

POTEC_EYETRACKING_DIR = POTEC_DIR / "eyetracking_data"


def _load_upstream_download_data():
    """Load ``download_data`` from the cloned PoTeC repo (init.sh clones it).

    Upstream resolves a relative ``output_folder`` against the script's own dir
    (``data/potec``), so passing ``"eyetracking_data"`` lands in
    ``POTEC_EYETRACKING_DIR`` with no fork needed.
    """
    path = POTEC_DIR / "download_data_files.py"
    spec = importlib.util.spec_from_file_location("potec_download_data_files", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.download_data


def download_eyetracking() -> None:
    print(f"Downloading PoTeC eye-tracking data -> {POTEC_EYETRACKING_DIR} ...")
    download_data = _load_upstream_download_data()
    # merged reading measures + scanpaths, fixations, raw data (upstream defaults).
    download_data(
        extract=True,
        output_folder="eyetracking_data",
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
