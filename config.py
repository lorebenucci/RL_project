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
TYPE_ATOMS=['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'K','Cu', 'In', 'Sn', 'Mn', 'Zn', 'Bi', 'Pt', 'Be', 'Li', 'V', 'Hg', 'Se', 'Pd', 'Pb', 'Zr', 'Yb', 'Nd', 'Cd', 'Sb', 'Ti', 'Ag', 'Dy', 'Ni', 'Gd', 'Au', 'Ba', 'Cr', 'Ge', 'Sr', 'Mo', 'Tl', 'H', 'Co']
HYBRIDIZATION_TYPE = [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED]
NUMBER_HYDROGENS=[0, 1, 2, 3, 4]
ATOM_DEGREE=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
PERMITTED_BONDS = [
        Chem.rdchem.BondType.SINGLE, 
        Chem.rdchem.BondType.DOUBLE, 
        Chem.rdchem.BondType.TRIPLE, 
        Chem.rdchem.BondType.AROMATIC
    ]
STEREO_LIST = [
        Chem.rdchem.BondStereo.STEREONONE, 
        Chem.rdchem.BondStereo.STEREOE, 
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
CHIRALITY_LIST = ['R', 'S', 'None']


TRAIN_PERCENTAGE=0.7
VAL_PERCENTAGE=0.15
TEST_PERCENTAGE=0.15

BATCH_SIZE =64
PRE_LOAD_BATCH=4
PIN_MEMORY=True
#---------------
#--- GNN MODEL CONFIG ---

EPOCHS = 200
HIDDEN_CHANNELS=128 #number of channels 
NUM_CLASSES=12
NODE_FEATURES=88
DROPOUT=0.41573689676626024
LR =0.006153085601625313
WEIGHT_DECAY = 0.00026568139241144923
# ------------------------
NUM_GLOBAL_FEATURES=2

LATENT_DIM = (HIDDEN_CHANNELS * 5 * 2) + NUM_GLOBAL_FEATURES


# --- RL AGENT CONFIG
GAMMA= 0.99 #discount factor
BATCH_SIZE_RL_AGENT=128
LR_GENERATOR= 3e-4

EPOCHS_AGENT = 3000
MEMORY_SIZE_BUFFER= 20000 # Replay buffer
MAX_STEPS =1 # number of possible modifies for each episode
DIM_DESCRIPTORS=2048

TAU_START=0.005
TAU_MIN=0.001
TAU_DECAY=0.995
ALPHA=0.6


EPSILON_START =1.0 # Initial exploration (max exploration)
EPSILON_MIN = 0.05 # Final exploration 
EPSILON_DECAY = 0.9985  #epsilon decay 

#----------------------------

# ---------------------------
#Measurement for MW AND LOGP
MEAN_MW=276.1441553893223
STD_MW=164.73235552463632
MEAN_LOGP=2.373942812220377
STD_LOGP=2.3043073032948307