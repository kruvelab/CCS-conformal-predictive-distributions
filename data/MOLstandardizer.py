import rdkit
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit import RDLogger
# all logs disabled except errors
# RDLogger.DisableLog('rdApp.info')
# RDLogger.DisableLog('rdApp.logIt')
# RDLogger.DisableLog('rdApp.setLevel')
# RDLogger.DisableLog('rdApp.debug')
# RDLogger.DisableLog('rdApp.critical')
# RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.*')



def MOLstandardizer(input, input_type = 'MOL',
                    stereochemistry = True, 
                    mol = True, smiles = False, inchikey = False, inchi = False, 
                    molecular_formula = False, monoisotopic_mass = False):
    '''
    Standardize a molecular structure.
    Loosely based on: 
    https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    Additional comments taken from:
    https://www.mail-archive.com/rdkit-discuss@lists.sourceforge.net/msg10668.html
     
    Parameters
    ----------
    input : MOl file, str
        Object representing the molecular structure.
        It can be either a MOL file, an InChI or SMILES strings.
    input_type : {'MOL', 'SMILES', 'InChI'}, default = 'MOL'
    stereochemistry : bool, default = True
         Whether to remove stereochemistry or preserve it.
    mol, smiles, inchikey, inchi, molecular_formula, monoisotopic_mass : bool, default = {True, False, False, False, False, False}
        Whether to return the given type of information.

    Returns
    -------
    standardized_info : dict
        Python dictionary with possible keys 
        {'Absolute SMILES', 'Unique SMILES', 'InChI', 'InChIKey', 'MOL', 'Molecular Formula', 'Monoisotopic Mass'}.
    '''

    if input_type == 'MOL':
        molecule = Chem.Mol(input) # better than copy.deepcopy(mol) https://sourceforge.net/p/rdkit/mailman/message/33652439/
    elif input_type == 'SMILES':
        molecule = Chem.MolFromSmiles(input) # generate molecule with sanitizer on
    elif input_type == 'InChI':
        molecule = Chem.MolFromInchi(input) # explicit hydrogens are removed for better computational efficiency
    else:
        raise Exception(
            'Only {MOL, SMILES, InChI} are accepted input types.'
        )
    '''
    When reading molecules, it automatically does sanitation (SanitizeMol() is different from the others: 
    it does a small amount of normalization - fixing groups like nitro which are commonly drawn in a
    hypervalent state but which can be represented in a charge-separated form without needing weird valences - and some validation - 
    rejecting molecules with atoms that have non-physical valences, rejecting molecules that cannot be kekulized - and a bunch of 
    chemistry perception - ring finding, calculating valences, finding aromatic systems, etc.)
    '''
    if molecule is None:
        raise Exception(
            'Input could not be copied/converted to a rdkit.Chem.rdchem.Mol file.'
        )

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
    
    molecular_information = {}
    if smiles:
        if stereochemistry:
            molecular_information['Absolute SMILES'] = Chem.MolToSmiles(molecule, isomericSmiles = True)
        else:
            molecular_information['Unique SMILES'] = Chem.MolToSmiles(molecule, isomericSmiles = False)
    if inchikey:
        molecular_information['InChIKey'] = Chem.MolToInchiKey(molecule)
    if inchi:
        molecular_information['InChI'] = Chem.MolToInchi(molecule)
    if molecular_formula:
        molecular_information['Molecular Formula'] = rdMolDescriptors.CalcMolFormula(molecule) # no need for explicit Hs
    if monoisotopic_mass:
        molecular_information['Monoisotopic Mass'] = Descriptors.ExactMolWt(molecule) # no need for explicit Hs
    if mol:
        if not stereochemistry:
            Chem.RemoveStereochemistry(molecule)
        molecular_information['MOL'] = molecule
    
    return molecular_information