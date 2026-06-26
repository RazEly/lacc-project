"""Config sanity: name maps and path wiring are internally consistent."""
from src import config


def test_et_measure_map_values_are_potec_columns():
    # plan-name -> PoTeC column mapping (memory: pipeline-decisions).
    assert config.ET_MEASURE_MAP["GD"] == "FPRT"
    assert config.ET_MEASURE_MAP["TRT"] == "TFT"
    assert config.ET_MEASURE_MAP["GPT"] == "RPD_inc"
    assert config.ET_MEASURE_MAP["F"] == "TFC"
    # identity-mapped measures.
    assert config.ET_MEASURE_MAP["FFD"] == "FFD"
    assert config.ET_MEASURE_MAP["SFD"] == "SFD"


def test_pca_measures_are_subset_of_mapped_measures():
    mapped = set(config.ET_MEASURE_MAP.values())
    assert set(config.PCA_MEASURES) <= mapped
    # PCA keeps the 4 informative measures, drops SFD + RPD_inc.
    assert "SFD" not in config.PCA_MEASURES
    assert "RPD_inc" not in config.PCA_MEASURES
    assert len(config.PCA_MEASURES) == 4


def test_grad_student_prompts_cover_both_domains():
    assert set(config.GRAD_STUDENT_PROMPTS) == {"physics", "biology"}
    assert all(p.strip() for p in config.GRAD_STUDENT_PROMPTS.values())


def test_paths_resolve_under_project_root():
    root = config.PROJECT_ROOT
    assert config.DATA_DIR.parent == root
    assert config.POTEC_WORD_FEATURES_DIR.is_relative_to(config.POTEC_DIR)
    assert config.POTEC_READING_MEASURES_DIR.is_relative_to(config.POTEC_DIR)
