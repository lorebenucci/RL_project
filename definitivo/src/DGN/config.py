import torch
from rdkit import Chem

# ==========================================
#  SYSTEM & RUNTIME 
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
NUM_WORKERS = 4
PIN_MEMORY = True
PRE_LOAD_BATCH = 4  

# Paths
DATASET_PATH = "../datasets/tox21_processed_features.csv"
MODEL_PATH = 'weights/DGN/best_GNN.pth'


# ==========================================
#  DATASET SPLIT 
# ==========================================
TRAIN_PERCENTAGE = 0.7
VAL_PERCENTAGE = 0.15
TEST_PERCENTAGE = 0.15
BATCH_SIZE = 64


# ==========================================
#  CHEMICAL FEATURES
# ==========================================
# Atom Properties
TYPE_ATOMS = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 
    'Al', 'I', 'B', 'K', 'Cu', 'In', 'Sn', 'Mn', 'Zn', 'Bi', 'Pt', 'Be', 'Li', 
    'V', 'Hg', 'Se', 'Pd', 'Pb', 'Zr', 'Yb', 'Nd', 'Cd', 'Sb', 'Ti', 'Ag', 'Dy', 
    'Ni', 'Gd', 'Au', 'Ba', 'Cr', 'Ge', 'Sr', 'Mo', 'Tl', 'H', 'Co'
]

HYBRIDIZATION_TYPE = [
    Chem.rdchem.HybridizationType.S, 
    Chem.rdchem.HybridizationType.SP, 
    Chem.rdchem.HybridizationType.SP2, 
    Chem.rdchem.HybridizationType.SP3, 
    Chem.rdchem.HybridizationType.SP3D, 
    Chem.rdchem.HybridizationType.SP3D2, 
    Chem.rdchem.HybridizationType.UNSPECIFIED
]

NUMBER_HYDROGENS = [0, 1, 2, 3, 4]
ATOM_DEGREE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
CHIRALITY_LIST = ['R', 'S', 'None']

# Bond Properties
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


# ==========================================
#  MODEL ARCHITECTURE
# ==========================================
NODE_FEATURES = 88
NUM_CLASSES = 12
NUM_GLOBAL_FEATURES = 2
HIDDEN_CHANNELS = 128

LATENT_DIM = (HIDDEN_CHANNELS * 5 * 2) + NUM_GLOBAL_FEATURES


# ==========================================
#  TRAINING
# ==========================================
EPOCHS = 200
LR = 0.006153085601625313
DROPOUT = 0.41573689676626024
WEIGHT_DECAY = 0.00026568139241144923