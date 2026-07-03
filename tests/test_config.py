"""Config sanity: name maps and path wiring are internally consistent."""
from src import config


def test_prior_config_present():
    # prompted arm uses a prior-reading passage + a length budget, not a persona.
    assert isinstance(config.PRIOR_MAX_TOKENS, int) and config.PRIOR_MAX_TOKENS > 0
    assert config.PRIOR_PASSAGE_SENTENCES > 0
    # neutral condition is averaged over >= N_PRIOR_PASSAGES distinct passages.
    assert config.N_PRIOR_PASSAGES > 0
    assert len(config.NEUTRAL_PRIORS) >= config.N_PRIOR_PASSAGES
    assert all(p.strip() for p in config.NEUTRAL_PRIORS)
    # persona prompts are gone.
    assert not hasattr(config, "GRAD_STUDENT_PROMPTS")


def test_paths_resolve_under_project_root():
    root = config.PROJECT_ROOT
    assert config.DATA_DIR.parent == root
    assert config.POTEC_WORD_FEATURES_DIR.is_relative_to(config.POTEC_DIR)
    assert config.POTEC_READING_MEASURES_DIR.is_relative_to(config.POTEC_DIR)
