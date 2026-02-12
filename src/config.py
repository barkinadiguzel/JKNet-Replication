# GNN backbone
NUM_LAYERS = 6
HIDDEN_DIM = 64
ACTIVATION = "relu"
DROPOUT = 0.5

JK_MODE = "concat"   # options: concat, maxpool, lstm

# Graph normalization
USE_NORMALIZATION = True
SELF_LOOPS = True

# Readout
READOUT_TYPE = "node"  # node or graph level
