
from rdkit import Chem
from rdkit.Chem import AllChem,rdFMCS, QED, Crippen, Descriptors
import py3Dmol
import webbrowser
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
import numpy as np

def compare_4_molecules_diff_highlight(smiles_list, names_list=None):
    """
    Visualizza 4 molecole (2 transizioni) evidenziando le differenze chimiche.
    """
    
    if names_list is None or len(names_list) != 4:
        names_list = ["Tossica (A)", "Non Tossica (A)", "Non Tossica (B)", "Tossica (B)"]

    # Griglia 2x2
    view = py3Dmol.view(width=1200, height=900, viewergrid=(2, 2))

    def process_pair(smiles_start, smiles_end, row):
        """Processa una coppia di molecole (Start -> End) e le mette nella riga specifica"""
        
        # 1. Creazione Molecole
        mol_start = Chem.MolFromSmiles(smiles_start)
        mol_end = Chem.MolFromSmiles(smiles_end)
        
        if not mol_start or not mol_end:
            print(f"Errore nella generazione delle molecole alla riga {row}")
            return

        # Aggiunta Idrogeni (importante per il 3D)
        mol_start = Chem.AddHs(mol_start)
        mol_end = Chem.AddHs(mol_end)

        # 2. Generazione 3D
        AllChem.EmbedMolecule(mol_start, AllChem.ETKDG())
        AllChem.EmbedMolecule(mol_end, AllChem.ETKDG())
        try:
            AllChem.MMFFOptimizeMolecule(mol_start)
            AllChem.MMFFOptimizeMolecule(mol_end)
        except: pass

        # 3. Trova la parte comune (MCS) per capire cosa è cambiato
        # Usiamo solo atomi pesanti per il confronto strutturale per evitare confusione con gli H
        mcs = rdFMCS.FindMCS([mol_start, mol_end], 
                             completeRingsOnly=True, 
                             atomCompare=rdFMCS.AtomCompare.CompareAny) # CompareAny permette di allineare scheletri simili
        
        common_substructure = Chem.MolFromSmarts(mcs.smartsString)

        # 4. Trova gli indici degli atomi che SONO DIVERSI (non matchano la parte comune)
        match_start = mol_start.GetSubstructMatch(common_substructure)
        match_end = mol_end.GetSubstructMatch(common_substructure)

        # Indici di tutti gli atomi
        all_idxs_start = set(range(mol_start.GetNumAtoms()))
        all_idxs_end = set(range(mol_end.GetNumAtoms()))

        # Gli atomi diversi sono quelli TOTALI meno quelli COMUNI
        diff_start = list(all_idxs_start - set(match_start)) # Atomi persi
        diff_end = list(all_idxs_end - set(match_end))       # Atomi guadagnati/cambiati

        # 5. Allineamento (Opzionale ma consigliato): Ruota mol_end su mol_start
        if match_end and match_start:
             try:
                 AllChem.AlignMol(mol_end, mol_start, atomMap=list(zip(match_end, match_start)))
             except: pass

        # --- AGGIUNTA AL VIEWER ---
        
        # Funzione interna per aggiungere modello e stile
        def add_single_mol(mol, r, c, name, diff_indices, color_base, diff_color="0xFF00FF"):
            block = Chem.MolToMolBlock(mol)
            view.addModel(block, 'mol', viewer=(r, c))
            
            # Stile Base (tutta la molecola)
            view.setStyle({'stick': {'colorscheme': 'greenCarbon', 'radius': 0.15}}, viewer=(r, c))
            
            # Stile Evidenziato (solo atomi diversi)
            # Creiamo una lista di stili per gli atomi specifici
            if diff_indices:
                # Evidenzia gli atomi cambiati con sfere più grandi e colore diverso
                view.addStyle({'index': diff_indices}, 
                              {'stick': {'color': diff_color, 'radius': 5}, 
                               'sphere': {'color': diff_color, 'radius': 0.4}}, 
                              viewer=(r, c))
                
                # Aggiungi etichette (Labels) sugli atomi cambiati
                conf = mol.GetConformer()
                for idx in diff_indices:
                    atom = mol.GetAtomWithIdx(idx)
                    symbol = atom.GetSymbol()
                    # Evitiamo di etichettare tutti gli Idrogeni per non fare confusione, a meno che non siano fondamentali
                    if symbol != 'H': 
                        pos = conf.GetAtomPosition(idx)
                        view.addLabel(symbol, 
                                      {'position': {'x': pos.x, 'y': pos.y, 'z': pos.z}, 
                                       'backgroundColor': diff_color, 
                                       'fontColor': 'black'
                                       },
                                      viewer=(r, c))

            # Titolo Pannello
            bg_title = "#b93e3e" if "red" in color_base else "#6de26d"
            view.addLabel(name, 
                          {'position': {'x': -2, 'y': 5, 'z': 0}, 
                           'backgroundColor': bg_title, 
                           'fontColor': 'black'}, 
                          viewer=(r, c))

        # Pannello Sinistro (Start)
        # Differenze in GIALLO/ORO per indicare "questo sta per cambiare"
        add_single_mol(mol_start, row, 0, names_list[row*2], diff_start, 
                       'redCarbon' if row==0 else 'greenCarbon', 
                       diff_color="gold") 

        # Pannello Destro (End)
        # Differenze in MAGENTA per indicare "questo è il nuovo gruppo"
        add_single_mol(mol_end, row, 1, names_list[row*2+1], diff_end, 
                       'greenCarbon' if row==0 else 'redCarbon', 
                       diff_color="magenta")

    # --- ESECUZIONE ---
    # Riga 1: Tossica A -> Sicura A
    process_pair(smiles_list[0], smiles_list[1], 0)
    
    # Riga 2: Sicura B -> Tossica B
    process_pair(smiles_list[2], smiles_list[3], 1)

    # --- RENDER E SALVATAGGIO ---
    view.zoomTo()
    
    output_file = "Diff_Analysis_1.html"
    html_content = view._make_html()
    
    with open(output_file, "w") as f:
        f.write(html_content)
        
    print(f"Generato report visuale: {output_file}")
    
    # Percorso assoluto per apertura sicura
    #file_path = "file://" + os.path.realpath(output_file)
    webbrowser.open(output_file)






def analysis(stats):
    
    #PLOT IN 3D MOLECULE BEFORE->AFTER THE FLIPP
    #max no tox and maxtox
    max_notox= max(stats["success_flip_fromtox_to_notox"], key=lambda x: x[2])
    max_tox=max(stats['success_flip_from_notox_totox'], key=lambda x: x[2])
    
    smiles_input=[max_notox[0],max_notox[1],max_tox[0],max_tox[1]]
    print
    print("best_smiles found\n")
    print(f"{smiles_input[0]}+\n")
    print(f"{smiles_input[1]}+\n")
    print(f"{smiles_input[2]}+\n")
    print(f"{smiles_input[3]}+\n")
    
          
    labels = [
    "Pantetina (Originale)", "Fosforato della Pantetina (Modificata)", 
    "Acido Ioxaglico (Origine)", "Tio-Deiodurato (Modificata)"
    ]
    compare_4_molecules_diff_highlight(smiles_input, labels)
    print("Hybrid_similarity of Streptozocina(originale)",max_notox[2])
    print("Hybrid_similarity Tilosina (Origine)",max_tox[2])
    #Hybrid_similarity of Streptozocina(originale) 0.9146143350601197
    #Hybrid_similarity Tilosina (Origine) 0.9346017826687205
    #ANALYZE ACTION DITRIBUTION
    analyze_actions_distribution(stats)
    
    #ANALYZE PROPERTIES PRESERVATION
    analyze_property_preservation(stats)
    
    #ANALYZE SCAFFOLD PRESERVATION
    analyze_scaffold_preservation(stats)
    
    

def classify_transformation_action_based(smiles_start, smiles_end):
    """
    Classifica la trasformazione basandosi rigorosamente sulle azioni 
    disponibili in ChemicalActionSpace (Molecule_env_actions.py).
    """
    mol_start = Chem.MolFromSmiles(smiles_start)
    mol_end = Chem.MolFromSmiles(smiles_end)
    
    if not mol_start or not mol_end: return "Errore Parsing"

    # 1. Conta Atomi (Solo Pesanti, ignoriamo H per ora)
    atoms_start = Counter([atom.GetSymbol() for atom in mol_start.GetAtoms()])
    atoms_end = Counter([atom.GetSymbol() for atom in mol_end.GetAtoms()])
    
    # 2. Calcola Differenza Netta
    diff = atoms_end.copy()
    diff.subtract(atoms_start)
    # Rimuovi i conteggi a zero (atomi invariati)
    diff = {k: v for k, v in diff.items() if v != 0}
    
    # Liste di atomi aggiunti e rimossi
    added = [k for k, v in diff.items() if v > 0]
    removed = [k for k, v in diff.items() if v < 0]

    # --- LOGICA DI CLASSIFICAZIONE (Mapping sulle Azioni Reali) ---

    # CASO 1: Nessuna variazione di composizione atomica (diff è vuoto)
    # Corrisponde alle Azioni 27-29: Add Bond / Remove Bond
    if not diff:
        # Calcoliamo info sui legami e anelli
        bonds_start = mol_start.GetNumBonds()
        bonds_end = mol_end.GetNumBonds()
        
        rings_start = mol_start.GetRingInfo().NumRings()
        rings_end = mol_end.GetRingInfo().NumRings()
        
        # Helper per contare i tipi di legame (Single vs Double)
        def count_bond_types(mol):
            cnt = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 0, 
                   Chem.BondType.TRIPLE: 0, Chem.BondType.AROMATIC: 0}
            for b in mol.GetBonds():
                bt = b.GetBondType()
                if bt in cnt: cnt[bt] += 1
            return cnt

        bt_start = count_bond_types(mol_start)
        bt_end = count_bond_types(mol_end)

        # Sottocaso A: Aggiunta Legame (Azioni 27, 28)
        # In questo env, aggiungere un legame crea sempre un nuovo anello (Ciclizzazione)
        if bonds_end > bonds_start:
            # Controlliamo quale tipo è aumentato
            if bt_end[Chem.BondType.DOUBLE] > bt_start[Chem.BondType.DOUBLE]:
                return "Ciclizzazione (Aggiunta Legame DOPPIO)"
            elif bt_end[Chem.BondType.SINGLE] > bt_start[Chem.BondType.SINGLE]:
                return "Ciclizzazione (Aggiunta Legame SINGOLO)"
            else:
                return "Ciclizzazione (Aggiunta Legame AROM/ALTRO)"

        # Sottocaso B: Rimozione Legame (Azione 29)
        # In questo env, rimuovere un legame apre un anello (Deciclizzazione)
        elif bonds_end < bonds_start:
            return "Apertura Anello (Rimozione Legame)"
            
        

    # CASO 2: Solo Aggiunta (Nessuna rimozione)
    # Corrisponde alle Azioni 0-17: Add Atom
    if added and not removed:
        # Se è stato aggiunto 1 solo tipo di atomo (come previsto dall'env)
        if len(added) == 1:
            atom_type = added[0]
            count = diff[atom_type]
            if count == 1:
                return f"Aggiunta Atomo (+{atom_type})"
            else:
                return f"Aggiunta Multipla (+{count} {atom_type})" # Raro in 1 step

    # CASO 3: Sostituzione (1 entra, 1 esce)
    # Corrisponde alle Azioni 18-26: Substitute Atom
    if len(added) == 1 and len(removed) == 1:
        in_atom = added[0]
        out_atom = removed[0]
        
        # Check quantità (deve essere 1 a 1 per una singola azione)
        if diff[in_atom] == 1 and diff[out_atom] == -1:
            # Sottocategorie utili per l'analisi qualitativa
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

    # CASO 4: Anomalie o Frammentazione
    # L'environment ha un check "if len(Chem.GetMolFrags(rw_mol)) > 1: return None"
    # quindi teoricamente non dovremmo vedere rimozioni pure, ma gestiamo il caso.
    if removed and not added:
        return f"Rimozione Anomala (-{removed[0]})"

    return "Modifica Complessa (Multi-step o Anomalia)"

def analyze_actions_distribution(stats):
    
    # Raccogli tutte le coppie
    all_pairs = []
    
    # Uniamo le liste di successo
    for start, end, sim in stats.get('success_flip_fromtox_to_notox', []):
        all_pairs.append((start, end))
    for start, end, sim in stats.get('success_flip_from_notox_totox', []):
        all_pairs.append((start, end))

    if not all_pairs:
        print("Nessun dato da analizzare.")
        return

    # Classifica
    categories = [classify_transformation_action_based(s, e) for s, e in all_pairs]
    counts = Counter(categories)
    
    # Plot
    df = pd.DataFrame.from_dict(counts, orient='index', columns=['Count']).reset_index()
    df = df.rename(columns={'index': 'Azione Rilevata'}).sort_values('Count', ascending=False)
    
    plt.figure(figsize=(11, 7)) # Aumento leggermente la dimensione per leggere meglio le label
    sns.set_style("whitegrid") # Aggiunge una griglia leggera di sfondo per facilitare la lettura
    
    # 1. CAMBIO PALETTE: 
    # 'viridis' = Gradiente luminoso (Giallo/Verde/Viola)
    # 'Set2' = Colori pastello distinti
    # 'Spectral' = Arcobaleno
    ax = sns.barplot(data=df, y='Azione Rilevata', x='Count', palette='Set2', orient='h')
    
    # 2. AGGIUNTA VALORI SULLE BARRE
    # Iteriamo sui "container" (i gruppi di barre) per aggiungere l'etichetta
    for container in ax.containers:
        ax.bar_label(container, 
                     padding=5,       # Spazio tra la fine della barra e il numero
                     fontsize=11,     # Grandezza del font
                     fmt='%d',        # Formato intero (niente virgole)
                     fontweight='bold') # Opzionale: Grassetto

    plt.title("Quali azioni sceglie l'agente per il Flip?", fontsize=16)
    plt.xlabel("Numero di Occorrenze", fontsize=12)
    plt.ylabel("") # Rimuovo l'etichetta Y ridondante
    
    plt.tight_layout() # Evita che le etichette lunghe vengano tagliate
    plt.show()
    
    return df



#import SA_SCORE
try:
    
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    
    HAS_SASCORE = True
except ImportError:
    HAS_SASCORE = False
    print("Avviso: 'sascorer.py' non trovato. L'analisi SA Score sarà saltata (verrà usato solo QED/LogP).")


def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    props = {}
    
    # 1. QED (Drug-likeness) - Range [0, 1] (Alto è meglio)
    props['QED'] = QED.qed(mol)
    
    # 2. LogP (Lipofilia)
    props['LogP'] = Crippen.MolLogP(mol)
    
    # 3. TPSA (Topological Polar Surface Area)
    props['TPSA'] = Descriptors.TPSA(mol)
    
    # 4. SA Score (Synthesizability) - Range [1, 10] (Basso è meglio/più facile)
    if HAS_SASCORE:
        try:
            props['SA_Score'] = sascorer.calculateScore(mol)
        except:
            props['SA_Score'] = np.nan
    
    #.5 MW (Molecular Weight)
    props['MW'] = Descriptors.MolWt(mol)
    
    return props


def analyze_property_preservation(stats):
    """
    Analizza se le proprietà chimiche (QED, SA, LogP, MW) sono preservate 
    durante la trasformazione.
    """
    data = []
    
    # Raccogli tutte le coppie di successo
    lists_to_check = [
        ('Tox->Safe', stats.get('success_flip_fromtox_to_notox', [])),
        ('Safe->Tox', stats.get('success_flip_from_notox_totox', []))
    ]
    
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

    # --- PLOTTING ---
    # Setup: 2 o 3 colonne a seconda se abbiamo SA Score
    cols = 4 if HAS_SASCORE else 2
    fig, axes = plt.subplots(1, cols, figsize=(6*cols, 6))
    if cols == 1: axes = [axes] # Gestione singolo asse
    
    sns.set_style("whitegrid")

    # Funzione helper per disegnare diagonale e scatter
    def plot_diagonal_scatter(ax, x_col, y_col, title, limit_range=None):
        # Scatter plot
        sns.scatterplot(data=df, x=x_col, y=y_col, hue='Direction', 
                        style='Direction', alpha=0.7, ax=ax, palette='deep')
        
        # Diagonale (Identity Line x=y)
        min_val = min(df[x_col].min(), df[y_col].min())
        max_val = max(df[x_col].max(), df[y_col].max())
        
        # Buffer per estetica
        buff = (max_val - min_val) * 0.05
        lims = [min_val - buff, max_val + buff]
        
        if limit_range: lims = limit_range # Forza range (es. 0-1 per QED)
            
        ax.plot(lims, lims, ls="--", c=".3", label="Identità (Nessun Cambio)")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Originale (Start)", fontsize=12)
        ax.set_ylabel("Modificata (End)", fontsize=12)
        ax.legend(loc='lower right')

    #  QED PLOT
    plot_diagonal_scatter(axes[0], 'QED_Start', 'QED_End', "QED (Drug-likeness)\nPreservation", limit_range=[0,1])
    
    #  LogP PLOT
    plot_diagonal_scatter(axes[1], 'LogP_Start', 'LogP_End', "LogP (Lipophilicity)\nPreservation")
    
    
    # SA Score PLOT (Se disponibile)
    if HAS_SASCORE:
        plot_diagonal_scatter(axes[3], 'SA_Start', 'SA_End', "SA Score (Synthesizability)\n(Basso = Facile)")

    
    # MW PLOT
    plot_diagonal_scatter(axes[2], 'MW_Start' , 'MW_End' , "MW Score (Molecular weight)\n(Preservation)")
    
    
    plt.tight_layout()
    plt.show()
    
    # --- REPORT STATISTICO ---
    print("\n--- ANALISI PROPRIETÀ DRUG-LIKE ---")
    print(f"Totale transizioni analizzate: {len(df)}")
    
    # Calcoliamo quanti 'peggiorano' significativamente (> 0.1 di differenza)
    worsened_qed = len(df[df['QED_End'] < df['QED_Start'] - 0.1])
    print(f"Molecole con crollo QED (>0.1): {worsened_qed} / {len(df)} ({worsened_qed/len(df):.1%})")
    
    if HAS_SASCORE:
        # SA Score: Se aumenta, diventa più difficile (peggio)
        harder_synth = len(df[df['SA_End'] > df['SA_Start'] + 1.0])
        print(f"Molecole diventate difficili da sintetizzare (Delta SA > 1.0): {harder_synth} ({harder_synth/len(df):.1%})")

    # --- ANALISI PROPRIETÀ DRUG-LIKE ---
    #Totale transizioni analizzate: 332
    #Molecole con crollo QED (>0.1): 49 / 332 (14.8%)
    #Molecole diventate difficili da sintetizzare (Delta SA > 1.0): 103 (31.0%)
    

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
    """
    Confronta gli scaffold prima e dopo la modifica.
    """
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
            
            # Confronto stringhe SMILES degli scaffold
            is_preserved = (scaf_start == scaf_end)
            
            # Categorizzazione dettagliata
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
    
    # Ordiniamo per frequenza
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
    
    # --- REPORT TESTUALE ---
    n_total = len(df)
    n_preserved = df['Preserved'].sum()
    
    
    print(f"\n--- SCAFFOLD ANALYSIS REPORT ---")
    print(f"Totale Modifiche: {n_total}")
    print(f"Side Chain Mods (Scaffold Preservato): {n_preserved} ({n_preserved/n_total:.1%})")
    print(f"Core Mods (Scaffold Modificato):       {n_total - n_preserved} ({(n_total - n_preserved)/n_total:.1%})")
    
    #--- SCAFFOLD ANALYSIS REPORT ---
    #Totale Modifiche: 332
    #Side Chain Mods (Scaffold Preservato): 237 (71.4%)
    #Core Mods (Scaffold Modificato):       95 (28.6%)
    
    return df