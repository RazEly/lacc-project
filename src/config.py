"""Central paths and dataset identifiers. Paths resolve from the project root.

Knobs used by a single module live in that module (e.g. the DAPT hyper-parameter
tables in main, the PoTeC sub-paths in features.potec). This file holds only what
more than one module shares.
"""

from pathlib import Path

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# PoTeC (eye-tracking corpus). Sub-paths (word_features / reading measures /
# eyetracking) are derived where they're read — acquire.download_potec, features.potec.
POTEC_DIR = DATA_DIR / "potec"

# german-commons (fine-tuning corpus). The HF repo/config id lives in
# acquire.download_commons; the quality gate (OCR score, length) is applied in
# acquire.domain_preprocessing.apply_quality_filter.
COMMONS_DIR = DATA_DIR / "commons_scientific"

# The two DAPT domains. The full name is the ONLY spelling anywhere — dataset
# labels, cache-column suffixes, run dirs, data dirs — no shorthand aliases.
DOMAINS = ("physics", "biology")

DOMAIN_DIRS = {domain: DATA_DIR / f"domain_{domain}" for domain in DOMAINS}
# Neutral control pool: german-commons docs OUTSIDE the pass-1 candidate set
# (both domain TF-IDF sims below threshold) — same corpus and register as the
# domain pools, no physics/biology content. See
# acquire.domain_preprocessing.select_other.
DOMAIN_OTHER_DIR = DATA_DIR / "domain_other"

# Default decoder LM; loaders are model-agnostic within the decoder family.
DEFAULT_MODEL = "dbmdz/german-gpt2"

# Run artifacts — anchored to the project root like every other path, so runs
# from any cwd hit the same DAPT/surprisal caches (a cwd-relative path made a
# wrong-dir run silently retrain everything).
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
# Wide per-word surprisal cache: computed once, reloaded to skip model forwards.
# Delete the file (and the DAPT run dirs) by hand to force a recompute.
SURPRISAL_CACHE_PATH = ARTIFACTS_DIR / "surprisal.csv"

# Per-word join key for every word-level table.
WORD_KEY = ["text_id", "word_index_in_text"]

# Prompts averaged per prior condition. Each condition's surprisal is the mean
# per-word surprisal across this many DISTINCT priors (K distinct held-out docs
# per domain, K distinct neutral passages) so no single idiosyncratic passage
# drives the estimate. 5 sits at the variance-reduction elbow (1/sqrt(K) shrinks
# fast to K≈5, then flattens); raise for the final run if a K-sensitivity check
# still drifts. The off-domain pool (DOMAIN_OTHER_DIR) needs >= this many docs.
N_PRIOR_PASSAGES = 20


PHYSICS_SEED_TERMS = (
    # expert technical terms
    "Anregungszustände",
    "Bandlücke",
    "Brechzahlen",
    "dielektrische",
    "dielektrischen",
    "Dielektrizitätskonstante",
    "Dopplerverschiebung",
    "Elektronenparabel",
    "Emissionslinien",
    "emittierten",
    "Energieparabel",
    "Hadronen",
    "holografischen",
    "Hologramm",
    "Hologrammbild",
    "Hologrammplatte",
    "Hyperfeinstrukturenaufspaltungen",
    "Impulserhaltung",
    "Interferenzstrukturen",
    "Interferometrie",
    "Ionenquelle",
    "kinetische",
    "kinetischen",
    "Leitungsband",
    "Löcherparabel",
    "Lorentzkraft",
    "Mößbauerspektroskopie",
    "Parität",
    "Phasen",
    "phasenrichtig",
    "Phasenverschiebungen",
    "Phonon",
    "Photon",
    "Photonen",
    "Photonenabsorption",
    "Photonenenergie",
    "Photonenenergien",
    "Photonenimpuls",
    "Photonische",
    "Photons",
    "Quarkmodell",
    "Rekonstruktionswelle",
    "Spin",
    "Standardbänder",
    "Valenzband",
    "Valenzbandes",
    "Wellenamplitude",
    "Wellenvektor",
    "Zyklotron",
)
BIOLOGY_SEED_TERMS = (
    # expert technical terms
    "Agarose",
    "Agarose-Sieb",
    "Aktin",
    "Aktinfilamente",
    "Aktinfilaments",
    "Amyloid",
    "amyloiden",
    "amyloider",
    "Angiospermenblatt",
    "ATP",
    "ATP-Bindungsstelle",
    "Autokatalyse",
    "Banden",
    "bipolares",
    "Calmodulinkette",
    "Cellulose-Mikrofibrillen",
    "Cellulose-Synthasekomplexe",
    "Chloroplast",
    "Diffusionskoeffizienten",
    "Dimer",
    "endogenen",
    "Endonuklease-Schnitte",
    "Ethidiumbromid",
    "Ethidiumbromids",
    "Eukaryoten",
    "Filament",
    "Gelelektrophorese",
    "homologe",
    "homologen",
    "Hydrolyse",
    "II",
    "infektiöse",
    "Kongorot",
    "kontraktilen",
    "Ligation",
    "Matrix",
    "Matrize",
    "membranständige",
    "Mikrotubuli",
    "Motordomäne",
    "Myosin",
    "Myosine",
    "Nitrationen",
    "Parenchymzelle",
    "Phosphatreste",
    "Polymer",
    "Polymerasekettenreaktion-Produkte",
    "Primärstruktur",
    "Prion",
    "Prionenform",
    "Prioneninfektion",
    "Prionenprotein",
    "Prions",
    "Proteinaggregate",
    "Quartärstrukturen",
    "RecA",
    "RecA-Protein",
    "Rekombinationsreparatur",
    "rekombinatorischen",
    "Replikationsgabel",
    "Replikationskomplex",
    "Replikationslücke",
    "Schwesterchromatide",
    "Vesikel",
    "VI",
    "Wildtypprotein",
    "β-D-Glucose",
    "β-Faltblattstrukturen",
)
