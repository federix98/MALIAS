from pathlib import Path

RANDOM_STATE = 42

CONFIG_PATH = Path("./config/")
EVAL_PATH = Path("./results/eval")

META_PREDICTIONS = Path("./results/pred")

CHECKPOINTS_PATH = Path("results/checkpoints")

ABLATION_FEATURES_SELECT = Path("results/ablation/features")
ABLATION_FEATURES_PRED = Path("results/ablation/meta")
ABLATION_FEATURES_EVAL = Path("results/eval/ablation-features")


LOAD_FOLDER = Path("results/checkpoints")
GENTEST_RESULTS = Path("results/gentest/pred")
GENTEST_EVAL = Path("results/gentest/eval")