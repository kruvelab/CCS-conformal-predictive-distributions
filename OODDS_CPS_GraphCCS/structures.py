import numpy as np

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
# all logs disabled except errors
# RDLogger.DisableLog('rdApp.info')
# RDLogger.DisableLog('rdApp.logIt')
# RDLogger.DisableLog('rdApp.setLevel')
# RDLogger.DisableLog('rdApp.debug')
# RDLogger.DisableLog('rdApp.critical')
# RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.*')

import networkx as nx
from myopic_mces import apply_filter
from myopic_mces import MCES_ILP
from joblib import Parallel, delayed




def MOLstandardizer(molecular_identifier, input_type = 'SMILES'):
    '''
    Standardize a molecular structure.

    Code based on:
        - https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
            - Concepteual standardization pipeline steps steps were obtained.

    Additional comments taken from:
        - https://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg10668.html

    
    Parameters
    ----------
    molecular_identifier : str
        It can be either an InChI or a SMILES string.
    input_type : {'SMILES', 'InChI'}, default = 'SMILES'

        
    Returns
    -------
    dict
        Dictionary containing the standardized information with keys {'Absolute SMILES', 'InChI'}.
    '''
    try:
        if input_type == 'SMILES':
            molecule = Chem.MolFromSmiles(molecular_identifier) # generate molecule with sanitizer on
        elif input_type == 'InChI':
            molecule = Chem.MolFromInchi(molecular_identifier) # explicit hydrogens are removed for better computational efficiency
        '''
        When reading molecules, it automatically does sanitation (SanitizeMol() is different from the others: 
        it does a small amount of normalization - fixing groups like nitro which are commonly drawn in a
        hypervalent state but which can be represented in a charge-separated form without needing weird valences - and some validation - 
        rejecting molecules with atoms that have non-physical valences, rejecting molecules that cannot be kekulized - and a bunch of 
        chemistry perception - ring finding, calculating valences, finding aromatic systems, etc.)
        '''

        rdMolStandardize.CleanupInPlace(molecule)
        '''
        Cleanup does a number of standardization operations:
        - remove ¿certain? Hs
        - disconnect metal atoms
        - normalize the molecule. Normalization typically refers to making changes to molecules to get "functional groups" (loosely defined) 
        into a normal form.
        - reionize the molecule. Reionizing does two things:
            1. adds a charge to a small set of free atoms which are likely counterions. These include Na, Mg, Cl, etc.
                1a. if the above added a positive charge: remove an H from an acidic group to neutrailze the positive charge that was added.
            2. Moves negative charges from less acidic groups to more acidic groups.
        '''
        
        rdMolStandardize.FragmentParentInPlace(molecule) # part we are interested in (without solvent molecules, salts, ...)
        
        uncharger = rdMolStandardize.Uncharger()
        uncharger.unchargeInPlace(molecule) # neutralizing the molecule based on a prioritization ruleset 
        # (not necessarily chemically consistent)
        
        tautomers = rdMolStandardize.TautomerEnumerator()
        molecule = tautomers.Canonicalize(molecule) # enumerate all possible tautomers and retain one (not necessarily the most stable one)
        # Tautomer canonicalization can sometimes lead to loss of symmetry.

        return {'Absolute SMILES': Chem.MolToSmiles(molecule), 'InChI': Chem.MolToInchi(molecule)}


    except Exception:
        return {'Absolute SMILES': None, 'InChI': None}



def myopicMCESdistance(graph1, graph2, threshold):
    """
    Calculates the myopic MCES distance between two molecules.

    References: 
        - Coverage bias in small molecule machine learning, Kretschmer et al. (https://doi.org/10.1038/s41467-024-55462-w)

        
    Parameters
    ----------
    graph1 : networkx.classes.graph.Graph
        Graph of molecule 1 in the pair.
    graph2 : networkx.classes.graph.Graph
        Graph of molecule 2 in the pair.    
    threshold : float
        Threshold for the comparison. Exact distance is only calculated if the distance is lower than the threshold.

        
    Returns:
    -------
    float
        Myopic MCES distance.
    """
    # filter bounds
    try:
        bound, _ = apply_filter(graph1, graph2, threshold, always_stronger_bound=False)
        # avoid ILP result if bound strictly above threshold
        if bound > threshold:
            return bound
    except Exception:
        return np.nan

    # ILP exact calculation
    try:
        exact, _ = MCES_ILP(graph1, graph2, threshold, 
                            solver = 'default', solver_options={"msg": False, "threads": 1}, no_ilp_threshold=False)
        return exact
    except Exception :
        return np.nan
            


def myMCES(graph, background_graphs, threshold):
    """
    Calculates the myopic MCES distances between a molecule and the background.

    
    Parameters
    ----------
    graph : networkx.classes.graph.Graph
        Graph of the query molecule.
    background_graphs : list of networkx.classes.graph.Graph objects
        List of netkorkx graphs of the background molecules.
    threshold : float
        Threshold for the comparison. Exact distance is only calculated if the distance is lower than the threshold.

        
    Returns:
    -------
    list
        Distances between the molecules.
    """
    return [myopicMCESdistance(graph, background_graph, threshold) for background_graph in background_graphs]



def mol2graph(mol):
    '''
    Convert rdkit.Chem.rdchem Mol object to networkx.classes.graph.Graph.


    Parameters
    ----------
    mol : rdkit.Chem.rdchem Mol object


    Returns
    -------
    networkx.classes.graph.Graph
        Graph that represents the molecule.
        The bond types are represented as edge weights.
        The atom types are represented as atom attributes of the nodes.
    '''
    G=nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),atom=atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtom().GetIdx(),bond.GetEndAtom().GetIdx(),weight=bond.GetBondTypeAsDouble())
    return G



def construct_graph(molecular_identifiers):
    """ 
    Build a molecular graph from an InChI or SMILES strigs.

    Code based on:
        - https://github.com/AlBi-HHU/myopic-mces
            - Construction of the networkx.classes.graph.Graph object.
    
            
    Parameters
    ----------
    molecular_identifiers : pandas.Series of shape (2,)
        SMILES string ('Absolute SMILES') and InChI string ('InChI').

        
    Returns:
    -------
    networkx.classes.graph.Graph
        Graph that represents the molecule.
        The bond types are represented as edge weights.
        The atom types are represented as atom attributes of the nodes.
    """
    try:
        graph =  mol2graph(Chem.MolFromSmiles(molecular_identifiers['Absolute SMILES']))
    except Exception:
        try:
            graph = mol2graph(Chem.MolFromInchi(molecular_identifiers.InChI))
        except Exception:
            graph = None
    return graph



def myopicMCESruler(graphs, background_graphs, threshold = 10, n_jobs = 1, k = 25):
    '''
    Obtain kNN arithmetic mean myopic MCES distance for each input molecule.

    
    Parameters
    ----------
    graphs : list of networkx.classes.graph.Graph objects
    background_graphs : list of networkx.classes.graph.Graph objects
        Graphs of the training chemicals to calculate myopic MCES distances.
    threshold : float, default = 10
        Threshold for the comparison. Exact distance is only calculated if the distance is lower than the threshold.
    n_jobs : int, default = 1
        Number of parallel workers. -1 will use all available cores.
    k : int, default = 25
        Number of nearest neighbors considered to calculate the average.

        
    Returns
    -------
    numpy.ndarray of shape (n,)
        Arithmetic mean distances.
    '''
    # choose parallel or serial execution
    if n_jobs == 1:
        print('Calculating myopic MCES distances sequentially.')
        distances_list = [
            myMCES(g, background_graphs, threshold)
            for g in graphs
        ]
    else:
        print(f'Calculating myopic MCES distances in parallel.')
        distances_list = Parallel(
            n_jobs=n_jobs,
            batch_size='auto',
            pre_dispatch='2*n_jobs',
            verbose=6,
        )(delayed(myMCES)(g, background_graphs, threshold)
        for g in graphs)
    
    return np.mean(
        np.partition(np.asarray(distances_list, dtype = np.float16), kth = k, axis = 1)[:, :k], # first k are the lowest
        axis=1
    ).astype(np.float64)
