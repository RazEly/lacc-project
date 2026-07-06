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
    # domain corpora are the term-targeted Wikipedia scrape, not commons.
    assert all(d.name == f"wiki_{k}" for k, d in config.DOMAIN_DIRS.items())
    # no hand-written / persona prompts, no off-domain neutral pool.
    assert not hasattr(config, "NEUTRAL_PRIORS")
    assert not hasattr(config, "DOMAIN_OTHER_DIR")
    assert not hasattr(config, "GRAD_STUDENT_PROMPTS")


def test_paths_resolve_under_project_root():
    root = config.PROJECT_ROOT
    assert config.DATA_DIR.parent == root
    assert potec.POTEC_WORD_FEATURES_DIR.is_relative_to(config.POTEC_DIR)
    assert potec.POTEC_READING_MEASURES_DIR.is_relative_to(config.POTEC_DIR)
