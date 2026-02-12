
from rdkit import Chem
from rdkit.Chem import AllChem,rdFMCS, QED, Crippen, Descriptors
import py3Dmol
import webbrowser
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem.Scaffolds import MurckoScaffold

def compare_4_molecules_diff_highlight(smiles_list, names_list=None):
   
    
    if names_list is None or len(names_list) != 4:
        names_list = ["Tossica (A)", "Non Tossica (A)", "Non Tossica (B)", "Tossica (B)"]

    #GRID 2X2 for 2 flipped molecule (TOX(ORIGIN)->NOTOX(DEST) AND NOTOX(ORIGIN)->TOX(DEST))
    view = py3Dmol.view(width=1200, height=900, viewergrid=(2, 2))

    
    #Process a couple of molecule and it is placed on specific row
    def process_pair(smiles_start, smiles_end, row):
        
        mol_start = Chem.MolFromSmiles(smiles_start)
        mol_end = Chem.MolFromSmiles(smiles_end)
        
        if not mol_start or not mol_end:
            print(f"Errore nella generazione delle molecole alla riga {row}")
            return

        #ADD Hyrogens
        mol_start = Chem.AddHs(mol_start)
        mol_end = Chem.AddHs(mol_end)

        #3D Generated
        AllChem.EmbedMolecule(mol_start, AllChem.ETKDG())
        AllChem.EmbedMolecule(mol_end, AllChem.ETKDG())
        try:
            AllChem.MMFFOptimizeMolecule(mol_start)
            AllChem.MMFFOptimizeMolecule(mol_end)
        except: pass

        #(MCS) find similairty scaffold evaluating Number/Type of atoms 
        mcs = rdFMCS.FindMCS([mol_start, mol_end], 
                             completeRingsOnly=True, 
                             atomCompare=rdFMCS.AtomCompare.CompareAny) # CompareAny
        
        
        common_substructure = Chem.MolFromSmarts(mcs.smartsString)

        #match start and  match end of principal Molecule Scaffold 
        match_start = mol_start.GetSubstructMatch(common_substructure)
        match_end = mol_end.GetSubstructMatch(common_substructure)

        all_idxs_start = set(range(mol_start.GetNumAtoms()))
        all_idxs_end = set(range(mol_end.GetNumAtoms()))

        #compute diff atoms
        diff_start = list(all_idxs_start - set(match_start)) # Lost atoms
        diff_end = list(all_idxs_end - set(match_end))       # Gained atoms

        #Align Mole
        if match_end and match_start:
             try:
                 AllChem.AlignMol(mol_end, mol_start, atomMap=list(zip(match_end, match_start)))
             except: pass

        #PLot
        
        # add model molecule and style
        def add_single_mol(mol, r, c, name, diff_indices, color_base, diff_color="0xFF00FF"):
            block = Chem.MolToMolBlock(mol)
            view.addModel(block, 'mol', viewer=(r, c))
            
           
            view.setStyle({'stick': {'colorscheme': 'greenCarbon', 'radius': 0.15}}, viewer=(r, c))
            
            # Highlight style (only different atoms)
            if diff_indices:
                #highlights differents atoms and changed bond
                view.addStyle({'index': diff_indices}, 
                              {'stick': {'color': diff_color, 'radius': 5}, 
                               'sphere': {'color': diff_color, 'radius': 0.4}}, 
                              viewer=(r, c))
                
                # ADD LABELS TO THE ATOMS
                conf = mol.GetConformer()
                for idx in diff_indices:
                    atom = mol.GetAtomWithIdx(idx)
                    symbol = atom.GetSymbol()
        
                    pos = conf.GetAtomPosition(idx)
                    view.addLabel(symbol, 
                                      {'position': {'x': pos.x, 'y': pos.y, 'z': pos.z}, 
                                       'backgroundColor': diff_color, 
                                       'fontColor': 'black'
                                       },
                                      viewer=(r, c))

            #Panel title
            bg_title = "#b93e3e" if "red" in color_base else "#6de26d"
            view.addLabel(name, 
                          {'position': {'x': -2, 'y': 5, 'z': 0}, 
                           'backgroundColor': bg_title, 
                           'fontColor': 'black'}, 
                          viewer=(r, c))

        #Left Panel (Start)
        add_single_mol(mol_start, row, 0, names_list[row*2], diff_start, 
                       'redCarbon' if row==0 else 'greenCarbon', 
                       diff_color="gold") 

        #Right Panel (End)
        add_single_mol(mol_end, row, 1, names_list[row*2+1], diff_end, 
                       'greenCarbon' if row==0 else 'redCarbon', 
                       diff_color="magenta")

    #PROCESS + PLOT
    # Row 1: Tox A -> Safe A
    process_pair(smiles_list[0], smiles_list[1], 0)
    
    # row 2: Safe B -> TOX B
    process_pair(smiles_list[2], smiles_list[3], 1)

    
    view.zoomTo()
    
    output_file = "Diff_Analysis_Report.html"
    html_content = view._make_html()
    
    with open(output_file, "w") as f:
        f.write(html_content)
        
    print(f"Generato report visuale: {output_file}")
    
    # Percorso assoluto per apertura sicura
    webbrowser.open(output_file)






def analysis(stats):
    
    #PLOT IN 3D MOLECULE BEFORE->AFTER THE FLIP
    
    #max no tox and maxtox
    max_notox= max(stats["success_flip_fromtox_to_notox"], key=lambda x: x[2])
    max_tox=max(stats['success_flip_from_notox_totox'], key=lambda x: x[2])
    
    smiles_input=[max_notox[0],max_notox[1],max_tox[0],max_tox[1]]
    
          
    labels = [
    "Pantetina (Originale)", "Fosforato della Pantetina (Modificata)", 
    "Acido Ioxaglico (Origine)", "Tio-Deiodurato (Modificata)"
    ]
    compare_4_molecules_diff_highlight(smiles_input, labels)
    print("Hybrid_similarity of Streptozocina(originale)",max_notox[2])
    print("Hybrid_similarity Tilosina (Origine)",max_tox[2])
    
    #ANALYZE ACTION DITRIBUTION on all strict success molecules
    analyze_actions_distribution(stats)
    
    #ANALYZE PROPERTY PRESERVATION on all strict success molecules [qed,mw,sa_score,LogP]
    analyze_property_preservation(stats)
    
    #ANALYZE SCAFFOLD PRESERVATION on all strict success molecules
    analyze_scaffold_preservation(stats)
    
    


def classify_transformation_action_based(smiles_start, smiles_end):

    mol_start = Chem.MolFromSmiles(smiles_start)
    mol_end = Chem.MolFromSmiles(smiles_end)
    
    if not mol_start or not mol_end: return "Errore Parsing"

    #counts atoms
    atoms_start = Counter([atom.GetSymbol() for atom in mol_start.GetAtoms()])
    atoms_end = Counter([atom.GetSymbol() for atom in mol_end.GetAtoms()])
    
    #observe difference after the flip
    diff = atoms_end.copy()
    diff.subtract(atoms_start)
  
    diff = {k: v for k, v in diff.items() if v != 0}
    
    # List of added atoms and removed atoms
    added = [k for k, v in diff.items() if v > 0]
    removed = [k for k, v in diff.items() if v < 0]

   
    #MAPPING ACTIONS
    # Case 1: No variation of chemical composition
    #it is defined with application of Add Bond / Remove Bond
    if not diff:
        
        bonds_start = mol_start.GetNumBonds()
        bonds_end = mol_end.GetNumBonds()
        
        rings_start = mol_start.GetRingInfo().NumRings()
        rings_end = mol_end.GetRingInfo().NumRings()
        
        #Helper function to count type of links (Single vs Double)
        def count_bond_types(mol):
            cnt = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 0, 
                   Chem.BondType.TRIPLE: 0, Chem.BondType.AROMATIC: 0}
            for b in mol.GetBonds():
                bt = b.GetBondType()
                if bt in cnt: cnt[bt] += 1
            return cnt

        bt_start = count_bond_types(mol_start)
        bt_end = count_bond_types(mol_end)

        #SubCase 1: ADD Links creating a new ring
        if bonds_end > bonds_start:
           
            if bt_end[Chem.BondType.DOUBLE] > bt_start[Chem.BondType.DOUBLE]:
                return "Ciclizzazione (Aggiunta Legame DOPPIO)"
            elif bt_end[Chem.BondType.SINGLE] > bt_start[Chem.BondType.SINGLE]:
                return "Ciclizzazione (Aggiunta Legame SINGOLO)"
            else:
                return "Ciclizzazione (Aggiunta Legame AROM/ALTRO)"

        #SubCase 2: remotion of bond openining a cycle
        elif bonds_end < bonds_start:
            return "Apertura Anello (Rimozione Legame)"
            
    # CASE 2: adding an atom
    if added and not removed:
        if len(added) == 1:
            atom_type = added[0]
            count = diff[atom_type]
            if count == 1:
                return f"Aggiunta Atomo (+{atom_type})"
            else:
                return f"Aggiunta Multipla (+{count} {atom_type})" 

    # CASO 3: sostitution (1 atom <-> 1 atom)
    if len(added) == 1 and len(removed) == 1:
        in_atom = added[0]
        out_atom = removed[0]
        
      
        if diff[in_atom] == 1 and diff[out_atom] == -1:
            
            if out_atom == 'C' and in_atom == 'N': return "Bioisosteria (C -> N)"
            if out_atom == 'N' and in_atom == 'C': return "Bioisosteria (N -> C)"
            if out_atom == 'O' and in_atom == 'S': return "Bioisosteria (O -> S)"
            if out_atom == 'S' and in_atom == 'O': return "Bioisosteria (S -> O)"
            
            halogens = {'F', 'Cl', 'Br', 'I'}
            if in_atom in halogens and out_atom not in halogens:
                return f"Alogenazione ({out_atom} -> {in_atom})"
            if out_atom in halogens and in_atom not in halogens:
                return f"De-Alogenazione ({out_atom} -> {in_atom})"
                
            return f"Sostituzione ({out_atom} -> {in_atom})"

    # Last case in which we detect anomalies
    if removed and not added:
        return f"Rimozione Anomala (-{removed[0]})"

    return "Modifica Complessa (Multi-step o Anomalia)"

def analyze_actions_distribution(stats):
    
   
    all_pairs = []
    
    #take all list of success molecule (TOX->NOTOX AND NOTOX->TOX)
    for start, end, sim in stats.get('success_flip_fromtox_to_notox', []):
        all_pairs.append((start, end))
    for start, end, sim in stats.get('success_flip_from_notox_totox', []):
        all_pairs.append((start, end))

    if not all_pairs:
        print("Nessun dato da analizzare.")
        return

    
    categories = [classify_transformation_action_based(s, e) for s, e in all_pairs]
    counts = Counter(categories)
    
    # Plot
    df = pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).reset_index()
    df = df.rename(columns={'index': 'Azione Rilevata'}).sort_values('Count', ascending=False)
    
    plt.figure(figsize=(11, 7)) 
    sns.set_style("whitegrid")
    
    
    ax = sns.barplot(data=df, y='Azione Rilevata', x='Count', palette='Set2', orient='h')
    
    
    for container in ax.containers:
        ax.bar_label(container,padding=5,fontsize=11,fmt='%d',fontweight='bold') 

    plt.title("Quali azioni sceglie l'agente per il Flip?", fontsize=16)
    plt.xlabel("Numero di Occorrenze", fontsize=12)
    plt.ylabel("")
    
    plt.tight_layout() 
    plt.show()
    
    return df


#import SA_SCORE
try:
    from rdkit.Chem import RDConfig
    import os
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    HAS_SASCORE = True
except ImportError:
    HAS_SASCORE = False
    print("Avviso: 'sascorer.py' non trovato. L'analisi SA Score sarà saltata (verrà usato solo QED/LogP).")


def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    props = {}
    
    #QED (Drug-likeness) - Range [0, 1]
    props['QED'] = QED.qed(mol)
    
    #LogP (Lipofilia)
    props['LogP'] = Crippen.MolLogP(mol)
    
    # MW (Molecular Weight)
    props['MW'] = Descriptors.MolWt(mol)
    
    #SA Score (Synthesizability) - Range [1, 10]
    if HAS_SASCORE:
        try:
            props['SA_Score'] = sascorer.calculateScore(mol)
        except:
            props['SA_Score'] = np.nan

    return props


def analyze_property_preservation(stats):
   
    data = []
    
    lists_to_check = [
        ('Tox->Safe', stats.get('success_flip_fromtox_to_notox', [])),
        ('Safe->Tox', stats.get('success_flip_from_notox_totox', []))
    ]
    
    #Compute the properties [QED,LOGP,MW,SA_SCORES]
    for direction, pair_list in lists_to_check:
        for start, end, sim in pair_list:
            p_start = calculate_properties(start)
            p_end = calculate_properties(end)
            
            if p_start and p_end:
                entry = {
                    
                    'Direction': direction,
                    'Similarity': sim,
                    
                    'MW_Start': p_start['MW'],
                    'MW_End': p_end['MW'],
                    
                    # Start Props
                    'QED_Start': p_start['QED'],
                    'LogP_Start': p_start['LogP'],
                    
                    # End Props
                    'QED_End': p_end['QED'],
                    'LogP_End': p_end['LogP'],
                    
                    # Deltas
                    'Delta_QED': p_end['QED'] - p_start['QED'],
                }
                
                if HAS_SASCORE:
                    entry['SA_Start'] = p_start.get('SA_Score')
                    entry['SA_End'] = p_end.get('SA_Score')
                
                data.append(entry)

    if not data:
        print("Nessun dato sufficiente per l'analisi delle proprietà.")
        return

    df = pd.DataFrame(data)

    #PLOTTING 
    cols = 4 if HAS_SASCORE else 2
    fig, axes = plt.subplots(1, cols, figsize=(6*cols, 6))
    if cols == 1: axes = [axes] # Gestione singolo asse
    
    sns.set_style("whitegrid")

    # Funzione helper to draw diagonal
    def plot_diagonal_scatter(ax, x_col, y_col, title, limit_range=None):
        # Scatter plot
        sns.scatterplot(data=df, x=x_col, y=y_col, hue='Direction', 
                        style='Direction', alpha=0.7, ax=ax, palette='deep')
        
        # Diagonal(Identity Line x=y)
        min_val = min(df[x_col].min(), df[y_col].min())
        max_val = max(df[x_col].max(), df[y_col].max())
        
       
        buff = (max_val - min_val) * 0.05
        lims = [min_val - buff, max_val + buff]
        
        if limit_range: lims = limit_range 
            
        ax.plot(lims, lims, ls="--", c=".3", label="Identità (Nessun Cambio)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Originale (Start)", fontsize=12)
        ax.set_ylabel("Modificata (End)", fontsize=12)
        ax.legend(loc='lower right')

    #QED PLOT
    plot_diagonal_scatter(axes[0], 'QED_Start', 'QED_End', "QED (Drug-likeness)\nPreservation", limit_range=[0,1])
    
    #LogP PLOT
    plot_diagonal_scatter(axes[1], 'LogP_Start', 'LogP_End', "LogP (Lipophilicity)\nPreservation")
    
    
    # SA Score PLOT (Se disponibile)
    if HAS_SASCORE:
        plot_diagonal_scatter(axes[3], 'SA_Start', 'SA_End', "SA Score (Synthesizability)\n(Basso = Facile)")

    
    # MW PLOT
    plot_diagonal_scatter(axes[2], 'MW_Start' , 'MW_End' , "MW Score (Molecular weight)\n(Preservation)")
    
    
    plt.tight_layout()
    plt.show()
    
    # Statistical report
    print("\n--- Properties Analysis DRUG-LIKE ---")
    print(f"Totale transizioni analizzate: {len(df)}")
    
    #WORSENED QED for Molecules after toxicity flip
    worsened_qed = len(df[df['QED_End'] < df['QED_Start'] - 0.1])
    print(f"Molecole con crollo QED (>0.1): {worsened_qed} / {len(df)} ({worsened_qed/len(df):.1%})")
    
    if HAS_SASCORE:
        #SA Score
        harder_synth = len(df[df['SA_End'] > df['SA_Start'] + 1.0])
        print(f"Molecole diventate difficili da sintetizzare (Delta SA > 1.0): {harder_synth} ({harder_synth/len(df):.1%})")


def get_murcko_scaffold(smiles):
    "Estrae lo scaffold di Murcko"
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    try:
        # GetScaffoldForMol restituisce la molecola "nuda"
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False) # Ignoriamo stereochimica per lo scaffold base
    except:
        return None
       
def analyze_scaffold_preservation(stats):
    
    data = []
    pairs = stats.get('success_flip_fromtox_to_notox', []) + \
            stats.get('success_flip_from_notox_totox', [])
    
    if not pairs:
        print("Nessun dato per analisi scaffold.")
        return
    
    for start, end, _ in pairs:
        scaf_start = get_murcko_scaffold(start)
        scaf_end = get_murcko_scaffold(end)
        
        if scaf_start is not None and scaf_end is not None:
            
            #SMILES of the scaffold
            is_preserved = (scaf_start == scaf_end)
            
            
            if is_preserved:
                change_type = "Side Chain Modification"
            else:
                
                mol_s = Chem.MolFromSmiles(scaf_start)
                mol_e = Chem.MolFromSmiles(scaf_end)
                if mol_s.GetNumAtoms() == mol_e.GetNumAtoms():
                    change_type = "Scaffold Edit (Heteroatom Swap)"
                elif mol_s.GetNumAtoms() < mol_e.GetNumAtoms():
                    change_type = "Scaffold Expansion (New Ring/Linker)"
                else:
                    change_type = "Scaffold Contraction (Ring Opening)"

            data.append({
                'Start': start,
                'End': end,
                'Scaffold_Start': scaf_start,
                'Scaffold_End': scaf_end,
                'Type': change_type,
                'Preserved': is_preserved
            })

    
    df = pd.DataFrame(data)
    
    # --- PLOT ---
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    #Plot in frequency order 
    order = df['Type'].value_counts().index
    
    ax = sns.countplot(data=df, y='Type', order=order, palette='Set2', orient='h')
    
    # Labels
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontweight='bold')
        
    plt.title("Analisi Topologica: Dove agisce l'agente?", fontsize=15)
    plt.xlabel("Numero di Molecole",fontsize=12)
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
    
    # --- REPOR ---
    n_total = len(df)
    n_preserved = df['Preserved'].sum()
    
    
    print(f"\n--- SCAFFOLD ANALYSIS REPORT ---")
    print(f"Totale Modifiche: {n_total}")
    print(f"Side Chain Mods (Scaffold Preservato): {n_preserved} ({n_preserved/n_total:.1%})")
    print(f"Core Mods (Scaffold Modificato):       {n_total - n_preserved} ({(n_total - n_preserved)/n_total:.1%})")

    
    return df