"""TreeMoE-Spec: tree-aware MoE speculative decoding for Mixtral-8x7B + EAGLE-2."""

__version__ = "0.1.0"

# Mixtral-8x7B architecture constants (spec §1.1)
NUM_LAYERS = 32
NUM_EXPERTS = 8
TOP_K = 2
HIDDEN_DIM = 4096
INTERMEDIATE_DIM = 14336
VOCAB_SIZE = 32000

# Tree verification static shape (spec §2, CUDA Graph precondition)
TREE_SIZE = 64
TREE_MAX_DEPTH = 6
