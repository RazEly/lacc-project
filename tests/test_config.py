"""Config sanity: name maps and path wiring are internally consistent."""
from src import config


def test_grad_student_prompts_cover_both_domains():
    assert set(config.GRAD_STUDENT_PROMPTS) == {"physics", "biology"}
    assert all(p.strip() for p in config.GRAD_STUDENT_PROMPTS.values())


def test_paths_resolve_under_project_root():
    root = config.PROJECT_ROOT
    assert config.DATA_DIR.parent == root
    assert config.POTEC_WORD_FEATURES_DIR.is_relative_to(config.POTEC_DIR)
    assert config.POTEC_READING_MEASURES_DIR.is_relative_to(config.POTEC_DIR)
