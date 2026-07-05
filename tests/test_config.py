"""Config sanity: name maps and path wiring are internally consistent.

Single-module knobs live in their owning module (config.py's design), so the
prior/passage constants are imported from ``main`` / ``features.priors``.
"""
from src import config
from src.features import potec, priors


def test_prior_config_present():
    # prompted arm uses a prior-reading passage + a length budget, not a persona.
    assert priors.PRIOR_PASSAGE_SENTENCES > 0
    # every condition is averaged over N_PRIOR_PASSAGES distinct passages.
    assert config.N_PRIOR_PASSAGES > 0
    # neutral priors are sampled from the off-domain pool, not hand-written.
    assert not hasattr(config, "NEUTRAL_PRIORS")
    assert config.DOMAIN_OTHER_DIR.parent == config.DATA_DIR
    # persona prompts are gone.
    assert not hasattr(config, "GRAD_STUDENT_PROMPTS")


def test_paths_resolve_under_project_root():
    root = config.PROJECT_ROOT
    assert config.DATA_DIR.parent == root
    assert potec.POTEC_WORD_FEATURES_DIR.is_relative_to(config.POTEC_DIR)
    assert potec.POTEC_READING_MEASURES_DIR.is_relative_to(config.POTEC_DIR)
