#FILE OF SETTING PARAMETERS
import torch
from rdkit import Chem

# -- Runtime Settings --
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
PRE_LOAD_BATCH=4
RANDOM_SEED=42
DATASET_PATH="./datasets/tox21_processed_features.csv"


#-------------Dataset transformation from smiles to Graph: we define all possible subset of parameters to build node structure
TYPE_ATOMS=['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'K', 'Unknown']
HYBRIDIZATION_TYPE = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, 'Misc']
NUMBER_HYDROGENS=[0, 1, 2, 3, 4]
ATOM_DEGREE=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

TRAIN_PERCENTAGE=0.7
VAL_PERCENTAGE=0.15
TEST_PERCENTAGE=0.15

BATCH_SIZE = 128
PRE_LOAD_BATCH=4
PIN_MEMORY=True
#---------------
#--- GNN MODEL CONFIG ---

EPOCHS = 200
HIDDEN_CHANNELS=128
NUM_CLASSES=12
NODE_FEATURES=56
DROPOUT=0.29239999526259247
LR =  0.00012892223135435807
WEIGHT_DECAY = 0.0004062781901413316

# --- RL AGENT CONFIG
