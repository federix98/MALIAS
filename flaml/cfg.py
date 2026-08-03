from pathlib import Path

# generic
RANDOM_STATE = 42

# dataset
TEST_SIZE, VAL_SIZE = .2, .5

# experiment setting
WINDOW, CONTEXT = 100, 50
HORIZON = WINDOW - CONTEXT


# training and testing for RNN
EPOCHS = 500
BATCH_SIZE = 128
ES_PATIENCE = 20
ROP_PATIENCE = 20
ES_START_FROM_EPOCH = 20

# path
DATA_PATH = "data"
RES_PATH = Path("results")

MODELS_PATH = RES_PATH / "models"
PRED_PATH = RES_PATH / "pred"

FULLPRED_PATH = RES_PATH / "full_pred"

MODELS_ORDER = ["sNaive", "sMM", "SARIMA", "ETS", "Prophet", "FC-RNN", "LSTM", "GRU", "TimesFM", "Chronos"]