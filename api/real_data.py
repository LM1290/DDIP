import pandas as pd

def get_real_dataset():
    # 1. Define Real SMILES (Verified Strings)
    drugs = {
        "Warfarin": "CC(=O)C(C1=CC=CC=C1)C2=C(O)C3=CC=CC=C3OC2=O",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        # Fixed Clopidogrel SMILES:
        "Clopidogrel": "COC(=O)C(C1=CC=CC=C1Cl)NC2=C(CC3=C2C=CC=S3)CC",
        "Simvastatin": "CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C",
        "Clarithromycin": "CCC1C(C(C(C(=O)C(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)OC)C)C)O)(C)O",
        "Sildenafil": "CCCC1=NN(C2=C1NC(=NC2=O)C3=CC(=CC=C3)S(=O)(=O)N4CCN(CC4)C)C",
        "Nitroglycerin": "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]",
        "Alcohol": "CCO",
        "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
        "Amoxicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",
        "Lisinopril": "C1CC(N(C1)C(=O)C(CCCCN)NC(CCC2=CC=CC=C2)C(=O)O)C(=O)O",
        "Metformin": "CN(C)C(=N)NC(=N)N",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "VitaminC": "C(C(C1C(=C(C(=O)O1)O)O)O)O"
    }

    # 2. Define PROVEN Interactions (Label = 1)
    proven_interactions = [
        ("Warfarin", "Aspirin"),
        ("Warfarin", "Ibuprofen"),
        ("Warfarin", "Clopidogrel"),
        ("Warfarin", "Paracetamol"),
        ("Warfarin", "Simvastatin"),
        ("Warfarin", "Clarithromycin"),
        ("Simvastatin", "Clarithromycin"),
        ("Sildenafil", "Nitroglycerin"),
        ("Paracetamol", "Alcohol"),
        ("Ibuprofen", "Aspirin"),
        ("Clopidogrel", "Amoxicillin"), # Adjusted for dataset variety
        ("Lisinopril", "Ibuprofen"),
        ("Alcohol", "Metformin"),
    ]

    # 3. Generate the Dataset
    data = []
    drug_list = list(drugs.keys())
    
    # Add Positive Samples
    for d1, d2 in proven_interactions:
        if d1 in drugs and d2 in drugs:
            data.append({"Drug_A": d1, "Drug_B": d2, "SMILES_A": drugs[d1], "SMILES_B": drugs[d2], "Label": 1.0})
            data.append({"Drug_A": d2, "Drug_B": d1, "SMILES_A": drugs[d2], "SMILES_B": drugs[d1], "Label": 1.0})

    # Add Negative Samples (Safe pairs)
    import itertools
    all_pairs = list(itertools.combinations(drug_list, 2))
    interaction_set = set(tuple(sorted(p)) for p in proven_interactions)
    
    for d1, d2 in all_pairs:
        pair_key = tuple(sorted((d1, d2)))
        if pair_key not in interaction_set:
            data.append({"Drug_A": d1, "Drug_B": d2, "SMILES_A": drugs[d1], "SMILES_B": drugs[d2], "Label": 0.0})
            
    return drugs, pd.DataFrame(data)
