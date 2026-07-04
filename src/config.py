"""Central paths and dataset identifiers. Paths resolve from the project root.

Knobs used by a single module live in that module (e.g. the DAPT hyper-parameter
tables in main, the PoTeC sub-paths in features.potec). This file holds only what
more than one module shares.
"""

from pathlib import Path

import torch

# Single compute-device decision for the whole pipeline: pick CUDA when present,
# else CPU. Modules move models/inputs onto this and gate CUDA-only flags on it,
# so a run never splits work across devices.
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

# TF-IDF domain seed terms — the pass-1 labelling seed bags. These are the
# technical terms flagged in the PoTeC stimuli ``word_features`` annotations
# (expert ∪ general technical terms), as unique surface forms per text domain. A
# keyword bag, not prose — prose shares too many char n-grams with any German
# text and collapses the TF-IDF similarity spread. Consumed by
# acquire.domain_preprocessing.load_domain_seeds.
PHYSICS_SEED_TERMS = (
    # expert technical terms
    "Anregungszustände", "Bandlücke", "Brechzahlen", "dielektrische",
    "dielektrischen", "Dielektrizitätskonstante", "Dopplerverschiebung",
    "Elektronenparabel", "Emissionslinien", "emittierten", "Energieparabel",
    "Hadronen", "holografischen", "Hologramm", "Hologrammbild",
    "Hologrammplatte", "Hyperfeinstrukturenaufspaltungen", "Impulserhaltung",
    "Interferenzstrukturen", "Interferometrie", "Ionenquelle", "kinetische",
    "kinetischen", "Leitungsband", "Löcherparabel", "Lorentzkraft",
    "Mößbauerspektroskopie", "Parität", "Phasen", "phasenrichtig",
    "Phasenverschiebungen", "Phonon", "Photon", "Photonen", "Photonenabsorption",
    "Photonenenergie", "Photonenenergien", "Photonenimpuls", "Photonische",
    "Photons", "Quarkmodell", "Rekonstruktionswelle", "Spin", "Standardbänder",
    "Valenzband", "Valenzbandes", "Wellenamplitude", "Wellenvektor", "Zyklotron",
    # general technical terms
    "Ablenkfeld", "absorbieren", "absorbiertem", "Absorption",
    "Absorptionswahrscheinlichkeit", "Abszisse", "Anregung", "Anregungsenergie",
    "Atom", "Atome", "Beschleunigungsspannung", "effektive", "effektiven",
    "elektrisches", "Elektromagneten", "Elektron", "Elektronen", "Elektrons",
    "elementaren", "elementareren", "Elementarteilchen", "Emission",
    "emittieren", "emittiertem", "Energie", "Energiemaximum", "Energieminimum",
    "Feld", "freie", "Frequenz", "geladenen", "Geschwindigkeit",
    "hochenergetischen", "Hochfrequenz", "Hochfrequenzspannung", "Impuls",
    "Ionen", "Kristalle", "Kristalles", "Kristallgitter", "Kristalls",
    "Ladungen", "Lichtwelle", "Magnetfeld", "Magnetfeldes", "magnetische",
    "magnetisches", "Masse", "Medien", "Mikrozylinder", "Moment", "negativ",
    "negative", "neutrales", "Neutronen", "Neutrons", "Nukleonen", "optischen",
    "Ordinate", "Ordnungsschemas", "Parabeln", "Periode", "periodische",
    "periodischen", "Polarität", "Polen", "positiven", "Protonen", "Radien",
    "Radius", "Referenzwelle", "reflektiert", "reflektierten",
    "Reflexionsvermögen", "Rückstoß", "Rückstoßenergie", "Rückstoßimpuls",
    "Signalwelle", "Streuung", "Teilchen", "thermische", "transparente",
    "Übergänge", "Überlagerung", "Umlaufszeit", "Vakuumkammer", "Wellen",
    "Wellenlänge", "x-y-Ebene", "Zentripetalkraft", "z-Richtung",
    "zylindrischen", "π-vielfaches",
)
BIOLOGY_SEED_TERMS = (
    # expert technical terms
    "Agarose", "Agarose-Sieb", "Aktin", "Aktinfilamente", "Aktinfilaments",
    "Amyloid", "amyloiden", "amyloider", "Angiospermenblatt", "ATP",
    "ATP-Bindungsstelle", "Autokatalyse", "Banden", "bipolares",
    "Calmodulinkette", "Cellulose-Mikrofibrillen", "Cellulose-Synthasekomplexe",
    "Chloroplast", "Diffusionskoeffizienten", "Dimer", "endogenen",
    "Endonuklease-Schnitte", "Ethidiumbromid", "Ethidiumbromids", "Eukaryoten",
    "Filament", "Gelelektrophorese", "homologe", "homologen", "Hydrolyse", "II",
    "infektiöse", "Kongorot", "kontraktilen", "Ligation", "Matrix", "Matrize",
    "membranständige", "Mikrotubuli", "Motordomäne", "Myosin", "Myosine",
    "Nitrationen", "Parenchymzelle", "Phosphatreste", "Polymer",
    "Polymerasekettenreaktion-Produkte", "Primärstruktur", "Prion",
    "Prionenform", "Prioneninfektion", "Prionenprotein", "Prions",
    "Proteinaggregate", "Quartärstrukturen", "RecA", "RecA-Protein",
    "Rekombinationsreparatur", "rekombinatorischen", "Replikationsgabel",
    "Replikationskomplex", "Replikationslücke", "Schwesterchromatide",
    "Vesikel", "VI", "Wildtypprotein", "β-D-Glucose", "β-Faltblattstrukturen",
    # general technical terms
    "Apparat", "Auflösung", "Basen", "binden", "Biosphäre", "Calciumionen",
    "Cellulose", "chlorophyllfreien", "chlorophyllhaltige", "Chromosom",
    "codierenden", "Deletion", "diffundieren", "DNA-Fragmente", "DNA-Schaden",
    "DNA-Stelle", "Doppelbrechung", "Doppelhelices", "Doppelhelix", "einlagern",
    "Einzelstrang-DNA-Enden", "elektrischen", "+Ende", "Energie", "Enzyme",
    "Expression", "extrazellulär", "extrazelluläre", "Fadenform", "Fasern",
    "Feldes", "Gen", "Gens", "Glucose", "Hefen", "heterogen", "induzieren",
    "infektiöses", "Kette", "Ketten", "Kohlenhydrat", "komplementären",
    "Komplex", "Komplexes", "Konformationsänderung", "Kontraktionsmechanismus",
    "Ladung", "lineares", "Lösung", "Medium", "Membran", "mineralischer",
    "Mineralstoffe", "Mineralstoffgehalt", "Minuspol", "Molekül",
    "Motormoleküle", "Motorproteine", "Muskelzellen", "Nährstoffe", "negativ",
    "negative", "Nitrat", "Nitrataufnahme", "organisches", "Pflanzenkörper",
    "Pflanzenzellen", "Phosphat", "Phosphataufnahme", "Phosphatverarmungszonen",
    "Phosphatversorgung", "Photosynthese", "Pluspol", "polarisiertem",
    "Protein", "Proteine", "Proteinen", "Proteins", "Regulation", "Saccharose",
    "Sequenzen", "Speicherorganen", "Stärke", "synthetisiert", "Tochterstrang",
    "transpirierende", "Zelle", "Zellen", "zellulären",
)

# domain_preprocessing outputs — training-domain corpora by domain name
# (finetune loads these; priors adds "neutral").
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
N_PRIOR_PASSAGES = 5
