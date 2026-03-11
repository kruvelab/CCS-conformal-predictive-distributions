'''
Code needed to run GraphCCS model to perform CCS point estimation.

References:
    - Large-scale prediction of collision cross-section 
    with very deep graph convolutional network for small molecule identification, 
    Xie et al. (https://doi.org/10.1016/j.chemolab.2024.105177)

Code from:
    - https://github.com/tingxiecsu/GraphCCS, except:
        - New documentation
        - Minor changes into functions needed to transform molecular ID + Ion Type ---> Ion Graph.
        - New functions for loading GraphCCS and performing point estimation.
'''
from functools import partial

from tqdm import tqdm

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Lipinski, rdMolDescriptors

from sklearn.preprocessing import normalize

import dgl
from dgllife.utils import mol_to_bigraph, BaseBondFeaturizer, ConcatFeaturizer
from dgllife.utils import bond_type_one_hot, bond_is_in_ring_one_hot, bond_is_conjugated_one_hot, bond_stereo_one_hot, bond_direction_one_hot
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgllife.model.readout.attentivefp_readout import AttentiveFPReadout

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import data
from torch.utils.data import SequentialSampler




###########################################################
###########################################################
######### molecular ID + Ion Type ---> Ion Graph ##########
###########################################################
###########################################################
class OneHotEncoder():
    '''
    Turn categorical data (with unique categories) into one-hot-encoded vectors (0s and 1s).

    This constructor initializes the class.
    All key attributes are initialized by default (empty dictionary).


    Parameters
    ----------
    None


    Attributes
    ----------
    category_index : dict
        Dictionary that maps the category labels (keys) to its appearance index in the fitting input.
    '''
    def __init__(self):
        self.category_index = {}


    def fit(self, x):
        '''
        Save a dictionary that maps the category labels (keys) to its appearance index in the input.


        Parameters
        ----------
        x : iterable of length n_train


        Returns
        -------
        None
            Updates the following attributes:
                - category_index : dictionary that maps the category labels (keys) to its appearance index in the input.
        '''
        for i, j in enumerate(x):
            self.category_index[j] = i


    def transform(self, x):
        '''
        Transform a category label into a one-hot-encoded vector.


        Parameters
        ----------
        x : category to be one-hot-encoded


        Returns
        -------
        numpy.ndarray of shape (n_train,)
        '''
        Xencoded = np.zeros(len(self.category_index))
        try:
            Xencoded[self.category_index[x]] = 1
        except KeyError as e:
            raise ValueError(f'Unknown label: {x}.') from e
        
        return Xencoded



def atom2features(atom, mol):
    '''
    Calculates a set of 148 features for an atom in a molecule.


    Parameters
    ----------
    atom: rdkit.Chem.rdchem.Atom
    mol: rdkit.Chem.rdchem.Mol


    Returns
    -------
    numpy.ndarray of shape(148,)
    '''
    atom_index = atom.GetIdx()

    AllChem.ComputeGasteigerCharges(mol)
    GasteigerCharge = atom.GetProp('_GasteigerCharge')
    if GasteigerCharge in ['-nan', 'nan', '-inf', 'inf']:
       GasteigerCharge = 0
    
    contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    CripperLogP = contribs[atom_index][0]
    MolarRefrac = contribs[atom_index][1]

    atom_symbol_encoder = OneHotEncoder()
    atom_list = [
        'F', 'Bk', 'Ge', 'Ce', 'Mn', 'Zn', 'Fe', 'Th', 'Tb', 'H', 'Ac',
        'Kr', 'Dy', 'Cf', 'Cl', 'Nd', 'Pt', 'Co', 'Sb', 'Ne', 'Zr', 'S',
        'C', 'O', 'Pa', 'Ru', 'Eu', 'V', 'La', 'Bi', 'Se', 'Li', 'Si',
        'Ag', 'Au', 'Ti', 'Pm', 'He', 'Tl', 'Ra', 'Hf', 'Lu', 'Na', 
        'Sr', 'I', 'Al', 'Cm', 'Cs', 'Br', 'Mo', 'Er', 'Cu', 'Ts', 
        'Mg', 'Pb', 'Gd', 'Tc', 'Rh', 'Ir', 'Yb', 'Te', 'B', 'N', 'Po',
        'Pu', 'Hg', 'Ba', 'Np', 'Sc', 'As', 'At', 'K', 'Ca', 'Pd', 'P',
        'In', 'Rn', 'Be', 'Pr', 'Os', 'Cd', 'Sn', 'W', 'Xe', 'Ni', 'Ta',
        'Nb', 'Fr', 'Ar', 'Tm', 'Re', 'Rb', 'Sm', 'Cr', 'Ga', 'Am', 
        'Ho', 'Md'
    ]
    atom_symbol_encoder.fit(atom_list)
    atom_symbol = atom_symbol_encoder.transform(atom.GetSymbol())

    atom_is_in_ring_encoder = OneHotEncoder()
    atom_is_in_ring_encoder.fit([True,False])
    atom_ring = atom_is_in_ring_encoder.transform(atom.IsInRing())
    atom_aromatic = atom_is_in_ring_encoder.transform(atom.GetIsAromatic())
    mol = atom.GetOwningMol()
    atom_hacceptor =  atom_is_in_ring_encoder.transform(atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)])
    atom_hdonor = atom_is_in_ring_encoder.transform(atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)])

    atom_hybrid_encoder = OneHotEncoder()
    atom_hybrid_encoder.fit([
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
        Chem.rdchem.HybridizationType.S
    ])
    atom_hybrid = atom_hybrid_encoder.transform(atom.GetHybridization())

    atom_numhs_encoder = OneHotEncoder()
    atom_numhs_encoder.fit([0,1,2,3,4])
    atom_numhs = atom_numhs_encoder.transform(min(atom.GetTotalNumHs(),4))

    atom_valence_encoder=OneHotEncoder()
    atom_valence_encoder.fit([0,1,2,3,4,5,6])
    atom_valence=atom_valence_encoder.transform(min(atom.GetTotalValence(),6))

    atom_degree_encoder=OneHotEncoder()
    atom_degree_encoder.fit([0,1,2,3,4,5])
    atom_degree=atom_degree_encoder.transform(min(atom.GetDegree(),5))

    atom_ringsize_encoder=OneHotEncoder()
    atom_ringsize_encoder.fit([0,3,4,5,6,7,8,9,10])
    for ring_size in [10,9,8,7,6,5,4,3,0]:
        if atom.IsInRingSize(ring_size):
            break
    atom_ringsize = atom_ringsize_encoder.transform(ring_size)

    atom_formalcharge=float(atom.GetFormalCharge())

    atom_Chiral = float(atom.HasProp("_ChiralityPossible"))

    atom_mass = float(atom.GetMass()/100)

    asa = rdMolDescriptors._CalcLabuteASAContribs(mol)[0][atom_index] 

    tpsa = rdMolDescriptors._CalcTPSAContribs(mol)[atom_index]


    atom_features = np.concatenate(
        (atom_symbol,atom_degree,atom_ring,atom_hacceptor,
        atom_hdonor,atom_numhs,atom_aromatic,atom_hybrid,
        atom_valence,atom_ringsize),
    )

    # to normalize
    atom_features2normalize = [[atom_formalcharge,atom_Chiral,atom_mass,GasteigerCharge,CripperLogP,MolarRefrac,asa,tpsa]]
    atom_features2normalize = normalize(atom_features2normalize, axis=1, norm='max')
    atom_features2normalize = list(atom_features2normalize.squeeze(-2)) # first axis from list creation needs to be removed
    return np.concatenate((atom_features,atom_features2normalize))



def mol2atom_features(mol):
    '''
    Calculates a set of 148 features for each atom in a molecule.


    Parameters
    ----------
    mol : mol: rdkit.Chem.rdchem.Mol


    Returns
    -------
    dict
        Dictionary with key 'h' containing a torch.Tensor of shape (n_atom, 148),
        where n_atom is the number of atoms in mol.
    '''
    feature_tensor = []
    for atom in mol.GetAtoms():
        feature_tensor.append(atom2features(atom,mol))
    feature_tensor = np.array(feature_tensor).reshape(-1,148)
    feature_tensor = torch.tensor(feature_tensor)
    return {'h': feature_tensor.float()}



class BondFeaturizer(BaseBondFeaturizer): # for dgl, mol2bond_features
    def __init__(self, bond_data_field='e', self_loop=False):
        super(BondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer(
                [bond_type_one_hot,
                 bond_is_conjugated_one_hot,
                 bond_is_in_ring_one_hot,
                 bond_stereo_one_hot,
                 bond_direction_one_hot]
            )}, self_loop=self_loop)



def mol2ion(mol, ion_type):
    """
    Modify parent molecule to obtain ion structure.


    Parameters
    ----------
    mol : rdkit.Chem.rdchem Mol
    ion_type : str
        The following ion types are supported: 
        [M+H]+, [M+Na]+ and [M-H]-.


    Returns
    -------
    rdkit.Chem.rdchem Mol
        Modified parent molecule to represent the ion structure.
    """   
    n_atom = mol.GetNumAtoms()
    mol_noH = mol
    mol = Chem.AddHs(mol) # explicit hydrogens
    edit_mol = mol
    AllChem.ComputeGasteigerCharges(mol)
    partial_charge=[]
    for atom in mol.GetAtoms():
        GasteigerCharge = atom.GetProp('_GasteigerCharge')
        if GasteigerCharge in ['-nan', 'nan', '-inf', 'inf']:
           GasteigerCharge = 0
        partial_charge.append(float(GasteigerCharge))
    charge_ranking_idxs = [partial_charge.index(sorted(partial_charge[:n_atom])[-i]) for i in range(1,n_atom)] # np.argsort(-partial_charge[:n_atom]) ?
    min_charge_idx = partial_charge.index(min(partial_charge[:n_atom]))


    if ion_type == '[M+H]+':
        mod = Chem.MolFromSmiles('[H+]')
        combo = Chem.CombineMols(mol,mod)
        edcombo = Chem.EditableMol(combo)
        # form ionic bond between most negative heavy atom and H+
        edcombo.AddBond(min_charge_idx,len(partial_charge),order=Chem.rdchem.BondType.IONIC)
        edit_mol = edcombo.GetMol()

    elif ion_type == '[M+Na]+':
        mod = Chem.MolFromSmiles('[Na+]')
        combo = Chem.CombineMols(mol,mod)
        edcombo = Chem.EditableMol(combo)
        # form ionic bond between most negative heavy atom and Na+
        edcombo.AddBond(min_charge_idx,len(partial_charge),order=Chem.rdchem.BondType.IONIC)
        edit_mol = edcombo.GetMol()

    elif ion_type == '[M-H]-':
        H_idx = 0
        for charge_idx in charge_ranking_idxs:
            # remove a proton from the most electrophilic heavy atom
            if mol_noH.GetAtomWithIdx(charge_idx).GetTotalNumHs() != 0:
                atom = mol.GetAtomWithIdx(charge_idx)
                neighbors = [x for x in atom.GetNeighbors()]
                neighbors_idx = [x.GetAtomicNum() for x in atom.GetNeighbors()]
                if 1 in neighbors_idx:
                    H_idx = neighbors[neighbors_idx.index(1)].GetIdx()
                mw = Chem.RWMol(mol)
                mw.RemoveAtom(H_idx)
                edit_mol = mw.GetMol()
                break
            else:
                continue

    else:
        raise ValueError(f'Ion type {ion_type} not supported.')
    

    return edit_mol



def ion2graph(ion):
    '''
    Convert ions into molecular graphs needed for GraphCCS.

    
    Parameters
    ----------
    ion : pandas.Series of shape (3,)
        SMILES string ('Absolute SMILES'), InChI string ('InChI') and ion type ('Ion Type') for ions.

        
    Returns
    -------
    list of dgl.DGLGraph
        Computed molecular graphs for GraphCCS for each of the input ions.
    '''
    node_featurizer = mol2atom_features
    edge_featurizer = BondFeaturizer(bond_data_field='e', self_loop=True)
    fc = partial(mol_to_bigraph, add_self_loop=True)

    ion_type = ion['Ion Type']
    try:
        ion_graph = fc(
            mol = mol2ion(Chem.MolFromSmiles(ion['Absolute SMILES']), ion_type), 
            node_featurizer = node_featurizer, edge_featurizer = edge_featurizer, 
            explicit_hydrogens = True
        )
    except Exception:
        try:
            ion_graph = fc(
                mol = mol2ion(Chem.MolFromInchi(ion.InChI), ion_type), 
                node_featurizer = node_featurizer, edge_featurizer = edge_featurizer, 
                explicit_hydrogens = True
            )
        except Exception:
            ion_graph = None

        
    return ion_graph




###########################################################
###########################################################
#################### GraphCCS skeleton ####################
###########################################################
###########################################################
class LayerScale(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif 18 > depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        self.scale = nn.Parameter(torch.full((dim,), init_eps))


    def forward(self, x):
        return x* self.scale



class EmbeddingLayerConcat(nn.Module):
    def __init__(self, node_in_dim, node_emb_dim, edge_in_dim=None, edge_emb_dim=None):
        super(EmbeddingLayerConcat, self).__init__()
        self.node_in_dim = node_in_dim
        self.node_emb_dim= node_emb_dim
        self.edge_in_dim = edge_emb_dim
        self.edge_emb_dim=edge_emb_dim

        self.atom_encoder = nn.Linear(node_in_dim, node_emb_dim)
        if edge_emb_dim is not None:
            self.bond_encoder = nn.Linear(edge_in_dim, edge_emb_dim)


    def forward(self, g):
        node_feats, edge_feats= g.ndata["h"], g.edata["e"]
        node_feats = self.atom_encoder(node_feats)

        if self.edge_emb_dim is None:
            return node_feats
        else:
            edge_feats = self.bond_encoder(edge_feats)
            return  node_feats, edge_feats



class GCNLayerWithEdge(nn.Module):
    def __init__(self, in_feats, out_feats, depth,activation=None, dropout=0.,residual=True):
        super(GCNLayerWithEdge, self).__init__()

        self.activation = activation
        self.mlp = nn.Linear(in_feats, out_feats)
        self.dropout = nn.Dropout(dropout)
        self.ls2 = LayerScale(in_feats, depth)
        self.residual=residual


    def reset_parameters(self):
        self.graph_conv.reset_parameters()
        if self.residual:
            self.res_connection.reset_parameters()
        if self.bn:
            self.bn_layer.reset_parameters()


    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            g.ndata['h'] = node_feats
            g.edata['e'] = edge_feats
            g.apply_edges(fn.u_add_e('h', 'e', 'm'))

            g.edata['a'] = edge_softmax(g, g.edata['m'])
            g.update_all(lambda edge: {'x': edge.data['m'] * edge.data['a']},
                         fn.sum('x', 'm'))

            new_feats = g.ndata['m']
            new_feats = self.mlp(new_feats)
            new_feats = self.activation(new_feats)
            new_feats = self.dropout(new_feats)

            new_feats = self.ls2(new_feats) + node_feats

            return new_feats
        


class GraphCCS(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_feats=None, activation=F.relu,
                  dropout=0.1, gru_out_layer=2,residual=True):
        super(GraphCCS, self).__init__()

        if hidden_feats is None:
            hidden_feats = [200]*5

        in_feats = hidden_feats[0]
        n_layers = len(hidden_feats)

        activation = [activation for _ in range(n_layers)]
        dropout = [dropout for _ in range(n_layers)]

        lengths = [len(hidden_feats), len(activation),len(dropout)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats ' \
                                       'activation, and dropout to ' \
                                       'be the same, got {}'.format(lengths)

        self.embed_layer = EmbeddingLayerConcat(node_in_dim, hidden_feats[0], edge_in_dim, hidden_feats[0])
        self.hidden_feats = hidden_feats
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            depth=i
            self.gnn_layers.append(GCNLayerWithEdge(in_feats, hidden_feats[i], depth,activation[i],  dropout[i],residual=residual))
            in_feats = hidden_feats[i]

        self.readout = AttentiveFPReadout(
            hidden_feats[-1], num_timesteps=gru_out_layer, dropout=dropout[-1]
        )

        self.out = nn.Sequential(
            nn.Linear(hidden_feats[-1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )


    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()


    def forward(self, g):
        node_feat, edge_feat = self.embed_layer(g)
        for gnn in self.gnn_layers:
            node_feat = gnn(g, node_feat, edge_feat)
        hidden_feats = self.readout(g, node_feat)
        out = self.out(hidden_feats)
        return out




###########################################################
###########################################################
#################### Point estimation #####################
###########################################################
###########################################################
def loadGraphCCS(GraphCCSconfig, weights_path, device):
    '''
    Assemble the GraphCCS architecture and load the trained weights.


    Parameters
    ----------
    GraphCCSconfig : dict
        Dictionary containing GraphCCS settings.
    weights_path : str
        Path string indicating where the .pt file containing the trained GraphCCS weights are saved.
    device : torch.device
    
    
    Returns
    -------
    GraphCCS : torch.nn.Module
    '''
    # assemble architecture
    model=GraphCCS(
        node_in_dim=GraphCCSconfig['node_feat_size'], edge_in_dim=GraphCCSconfig['edge_feat_size'], 
        hidden_feats=[GraphCCSconfig['hid_dim']]*GraphCCSconfig['num_layers'],
        gru_out_layer=GraphCCSconfig['gru_out_layer'], 
        residual=True
    )

    # load trained weights
    state_dict = torch.load(weights_path, map_location = device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model



class infoData_retriever(data.Dataset):
    def __init__(self, ion_graphs):
        self.ion_graphs = ion_graphs


    def __len__(self):
        return len(self.ion_graphs)


    def __getitem__(self, index):
        return self.ion_graphs[index]



def batching(x):
	return dgl.batch(x)



def predict(GraphCCSconfig, ions, model, device):
    '''
    Use GraphCCS to predict the CCSs.


    Parameters
    ----------
    GraphCCSconfig : dict
        Dictionary containing GraphCCS settings.
    ions : list of length n of or dgl.DGLGraph
        Computed ion graphs for GraphCCS.
    model : torch.nn.Module
        GraphCCS.
    device : torch.device


    Returns
    -------
    y_hats : numpy.ndarray of shape (n,)
        Predicted CCSs.
    '''
    # create PyTorch data loader instance
    data_info = infoData_retriever(ions)
    loader_params = {
        'batch_size': GraphCCSconfig['batch_size'],
        'shuffle': False,
        'num_workers': GraphCCSconfig['num_workers'],
        'drop_last': False,
        'sampler':SequentialSampler(data_info),
        'collate_fn': batching
    }
    ion_loader = data.DataLoader(data_info, **loader_params)

    # forward pass
    y_hats = []
    model.eval() # turn-off drop-out and batch norm
    pbar = tqdm(total = len(ions), desc="CCS point estimation", unit = ' Ions')
    with torch.no_grad(): # turn-off autograd and speed it up
        for ion_graph_batch in ion_loader:
            ion_graph_batch = ion_graph_batch.to(device)
            batch_y_hats = model(ion_graph_batch)
            batch_y_hats = torch.squeeze(batch_y_hats).cpu().numpy() # detach() eliminated because grads already off
            y_hats = y_hats + batch_y_hats.flatten().tolist()
            pbar.update(batch_y_hats.size)
    pbar.close()
    return np.asarray(y_hats)
