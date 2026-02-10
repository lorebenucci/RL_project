import os
import sys
import webbrowser
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import py3Dmol
from src.DDQN.config import RESULTS_DIR

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, QED, Crippen, Descriptors, RDConfig
from rdkit.Chem.Scaffolds import MurckoScaffold

# Assicuriamoci che la directory di output esista
os.makedirs(RESULTS_DIR, exist_ok=True)

# Gestione import SA_Score
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    HAS_SASCORE = True
except ImportError:
    HAS_SASCORE = False
    print("Avviso: 'sascorer.py' non trovato. SA Score disabilitato.")

def save_plot(filename):
    """Helper per salvare i plot nella directory corretta."""
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato: {path}")
    plt.show()
    plt.close()

def compare_4_molecules_diff_highlight(smiles_list, names_list=None):
    """Genera report HTML 3D evidenziando le differenze."""
    if not names_list or len(names_list) != 4:
        names_list = ["Tossica (A)", "Non Tossica (A)", "Non Tossica (B)", "Tossica (B)"]

    view = py3Dmol.view(width=1200, height=900, viewergrid=(2, 2))

    def add_mol_view(mol, r, c, name, diff_idxs, color_scheme, highlight_color):
        block = Chem.MolToMolBlock(mol)
        view.addModel(block, 'mol', viewer=(r, c))
        view.setStyle({'stick': {'colorscheme': color_scheme, 'radius': 0.15}}, viewer=(r, c))
        
        if diff_idxs:
            view.addStyle({'index': diff_idxs}, 
                          {'stick': {'color': highlight_color, 'radius': 0.15}, 
                           'sphere': {'color': highlight_color, 'radius': 0.4}}, viewer=(r, c))
            
            conf = mol.GetConformer()
            for idx in diff_idxs:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetSymbol() != 'H':
                    pos = conf.GetAtomPosition(idx)
                    view.addLabel(atom.GetSymbol(), 
                                  {'position': {'x': pos.x, 'y': pos.y, 'z': pos.z}, 
                                   'backgroundColor': highlight_color, 'fontColor': 'black'}, viewer=(r, c))

        bg_title = "#b93e3e" if "red" in color_scheme else "#6de26d"
        view.addLabel(name, {'position': {'x': -2, 'y': 5, 'z': 0}, 
                             'backgroundColor': bg_title, 'fontColor': 'black'}, viewer=(r, c))

    def process_pair(s_start, s_end, row):
        mol_start = Chem.AddHs(Chem.MolFromSmiles(s_start))
        mol_end = Chem.AddHs(Chem.MolFromSmiles(s_end))
        if not mol_start or not mol_end: return

        AllChem.EmbedMolecule(mol_start, AllChem.ETKDG())
        AllChem.EmbedMolecule(mol_end, AllChem.ETKDG())
        try:
            AllChem.MMFFOptimizeMolecule(mol_start)
            AllChem.MMFFOptimizeMolecule(mol_end)
        except: pass

        mcs = rdFMCS.FindMCS([mol_start, mol_end], completeRingsOnly=True, atomCompare=rdFMCS.AtomCompare.CompareAny)
        common = Chem.MolFromSmarts(mcs.smartsString)
        match_start = mol_start.GetSubstructMatch(common)
        match_end = mol_end.GetSubstructMatch(common)
        
        diff_start = list(set(range(mol_start.GetNumAtoms())) - set(match_start))
        diff_end = list(set(range(mol_end.GetNumAtoms())) - set(match_end))

        if match_end and match_start:
            try: AllChem.AlignMol(mol_end, mol_start, atomMap=list(zip(match_end, match_start)))
            except: pass

        add_mol_view(mol_start, row, 0, names_list[row*2], diff_start, 'redCarbon' if row==0 else 'greenCarbon', "gold")
        add_mol_view(mol_end, row, 1, names_list[row*2+1], diff_end, 'greenCarbon' if row==0 else 'redCarbon', "magenta")

    process_pair(smiles_list[0], smiles_list[1], 0)
    process_pair(smiles_list[2], smiles_list[3], 1)

    view.zoomTo()
    output_filename = "Diff_Analysis_Report.html"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    
    with open(output_path, "w") as f:
        f.write(view._make_html())
    
    print(f"Report visuale salvato: {output_path}")
    # Usa path assoluto per compatibilità browser
    webbrowser.open("file://" + os.path.abspath(output_path))

def classify_transformation_action_based(smiles_start, smiles_end):
    mol_start = Chem.MolFromSmiles(smiles_start)
    mol_end = Chem.MolFromSmiles(smiles_end)
    if not mol_start or not mol_end: return "Errore Parsing"

    atoms_start = Counter([a.GetSymbol() for a in mol_start.GetAtoms()])
    atoms_end = Counter([a.GetSymbol() for a in mol_end.GetAtoms()])
    
    diff = atoms_end.copy()
    diff.subtract(atoms_start)
    diff = {k: v for k, v in diff.items() if v != 0}
    
    added = [k for k, v in diff.items() if v > 0]
    removed = [k for k, v in diff.items() if v < 0]

    if not diff:
        bonds_delta = mol_end.GetNumBonds() - mol_start.GetNumBonds()
        if bonds_delta > 0: return "Ciclizzazione (Aggiunta Legame)"
        elif bonds_delta < 0: return "Apertura Anello (Rimozione Legame)"
        return "Modifica Strutturale Neutra"

    if added and not removed:
        return f"Aggiunta Atomo (+{added[0]})" if len(added) == 1 else "Aggiunta Multipla"

    if len(added) == 1 and len(removed) == 1:
        in_a, out_a = added[0], removed[0]
        if diff[in_a] == 1 and diff[out_a] == -1:
            halogens = {'F', 'Cl', 'Br', 'I'}
            if in_a in halogens and out_a not in halogens: return f"Alogenazione ({out_a}->{in_a})"
            if out_a in halogens and in_a not in halogens: return f"De-Alogenazione ({out_a}->{in_a})"
            if (out_a, in_a) in [('C','N'), ('N','C'), ('O','S'), ('S','O')]: return f"Bioisosteria ({out_a}->{in_a})"
            return f"Sostituzione ({out_a}->{in_a})"

    return "Modifica Complessa/Anomala"

def analyze_actions_distribution(stats):
    all_pairs = stats.get('success_flip_fromtox_to_notox', []) + stats.get('success_flip_from_notox_totox', [])
    if not all_pairs: return

    categories = [classify_transformation_action_based(s, e) for s, e, _ in all_pairs]
    df = pd.DataFrame.from_dict(Counter(categories), orient='index', columns=['Count']).reset_index()
    df = df.rename(columns={'index': 'Azione'}).sort_values('Count', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df, y='Azione', x='Count', palette='Set2', orient='h')
    for container in ax.containers:
        ax.bar_label(container, padding=5, fontweight='bold')
    plt.title("Distribuzione Azioni Agente")
    
    save_plot("actions_distribution.png")

def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    props = {
        'QED': QED.qed(mol),
        'LogP': Crippen.MolLogP(mol),
        'MW': Descriptors.MolWt(mol),
        'SA_Score': sascorer.calculateScore(mol) if HAS_SASCORE else np.nan
    }
    return props

def analyze_property_preservation(stats):
    data = []
    directions = [('Tox->Safe', 'success_flip_fromtox_to_notox'), 
                  ('Safe->Tox', 'success_flip_from_notox_totox')]
    
    for label, key in directions:
        for start, end, sim in stats.get(key, []):
            p_s = calculate_properties(start)
            p_e = calculate_properties(end)
            if p_s and p_e:
                entry = {'Direction': label, 'Similarity': sim}
                for k in p_s:
                    entry[f'{k}_Start'] = p_s[k]
                    entry[f'{k}_End'] = p_e[k]
                data.append(entry)

    if not data: return
    df = pd.DataFrame(data)

    cols = 4 if HAS_SASCORE else 3
    fig, axes = plt.subplots(1, cols, figsize=(5*cols, 5))
    if cols == 1: axes = [axes]
    
    metrics = [('QED', [0,1]), ('LogP', None), ('MW', None)]
    if HAS_SASCORE: metrics.append(('SA_Score', None))

    for i, (metric, lims) in enumerate(metrics):
        ax = axes[i]
        sns.scatterplot(data=df, x=f'{metric}_Start', y=f'{metric}_End', hue='Direction', style='Direction', ax=ax)
        
        min_v = min(df[f'{metric}_Start'].min(), df[f'{metric}_End'].min())
        max_v = max(df[f'{metric}_Start'].max(), df[f'{metric}_End'].max())
        ax.plot([min_v, max_v], [min_v, max_v], ls="--", c=".3")
        
        if lims: ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title(f"{metric} Preservation")

    plt.tight_layout()
    save_plot("property_preservation.png")

    print("\n--- STATISTICHE PROPRIETÀ ---")
    print(f"Totale: {len(df)}")
    print(f"Crollo QED (>0.1): {len(df[df['QED_End'] < df['QED_Start'] - 0.1])}")
    if HAS_SASCORE:
        print(f"Peggioramento SA (>1.0): {len(df[df['SA_Score_End'] > df['SA_Score_Start'] + 1.0])}")

def get_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=False)
    except: return None

def analyze_scaffold_preservation(stats):
    pairs = stats.get('success_flip_fromtox_to_notox', []) + stats.get('success_flip_from_notox_totox', [])
    data = []

    for start, end, _ in pairs:
        s_start, s_end = get_murcko_scaffold(start), get_murcko_scaffold(end)
        if s_start and s_end:
            if s_start == s_end:
                ctype = "Side Chain Modification"
            else:
                ms, me = Chem.MolFromSmiles(s_start), Chem.MolFromSmiles(s_end)
                if ms.GetNumAtoms() < me.GetNumAtoms(): ctype = "Scaffold Expansion"
                elif ms.GetNumAtoms() > me.GetNumAtoms(): ctype = "Scaffold Contraction"
                else: ctype = "Scaffold Edit (Heteroatom)"
            
            data.append({'Type': ctype, 'Preserved': s_start == s_end})

    if not data: return
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    ax = sns.countplot(data=df, y='Type', order=df['Type'].value_counts().index, palette='Set2')
    for c in ax.containers: ax.bar_label(c, padding=3)
    plt.title("Analisi Modifiche Scaffold")
    
    save_plot("scaffold_analysis.png")

    print(f"\n--- SCAFFOLD REPORT ---")
    print(f"Preservati: {df['Preserved'].sum()} ({df['Preserved'].mean():.1%})")
    print(f"Modificati: {len(df) - df['Preserved'].sum()}")

def analysis(stats):
    try:
        max_notox = max(stats.get("success_flip_fromtox_to_notox", []), key=lambda x: x[2])
        max_tox = max(stats.get("success_flip_from_notox_totox", []), key=lambda x: x[2])
        
        smiles_input = [max_notox[0], max_notox[1], max_tox[0], max_tox[1]]
        labels = ["Start (Tox)", "End (Safe)", "Start (Safe)", "End (Tox)"]
        
        print(f"Best Flip Safe->Tox Sim: {max_tox[2]:.4f}")
        print(f"Best Flip Tox->Safe Sim: {max_notox[2]:.4f}")
        
        compare_4_molecules_diff_highlight(smiles_input, labels)
    except ValueError:
        print("Dati insufficienti per la visualizzazione 3D.")

    analyze_actions_distribution(stats)
    analyze_property_preservation(stats)
    analyze_scaffold_preservation(stats)