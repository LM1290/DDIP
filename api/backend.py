import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from fastapi import FastAPI
from pydantic import BaseModel
from real_data import get_real_dataset

app = FastAPI()

# ==========================================
# 1. CONFIG & MODEL ARCHITECTURE
# ==========================================
N_BITS = 1024
INPUT_DIM = N_BITS * 2

class DDI_Network(nn.Module):
    def __init__(self):
        super(DDI_Network, self).__init__()
        self.layer1 = nn.Linear(INPUT_DIM, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(512, 128)
        self.output = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.relu(self.layer2(x))
        return self.sigmoid(self.output(x))

# ==========================================
# 2. HELPER FUNCTIONS (Clean Version)
# ==========================================
# Initialize the generator once to avoid Deprecation Warnings
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)

def smiles_to_vector(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(N_BITS)
    
    # Use the new generator API
    fp = morgan_gen.GetFingerprint(mol)
    return np.array(fp)

# ==========================================
# 3. INITIALIZATION & TRAINING
# ==========================================
print("Initializing BioGuard... Loading Curated Interaction Database...")

drug_db, df_train = get_real_dataset()
print(f"Training on {len(df_train)} clinically validated pairs...")

X_data = []
y_data = []

for _, row in df_train.iterrows():
    v1 = smiles_to_vector(row['SMILES_A'])
    v2 = smiles_to_vector(row['SMILES_B'])
    combined = np.concatenate([v1, v2])
    X_data.append(combined)
    y_data.append(row['Label'])

X_train = torch.FloatTensor(np.array(X_data))
y_train = torch.FloatTensor(np.array(y_data)).unsqueeze(1)

model = DDI_Network()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

model.eval()
print(f"Model Converged. Final Loss: {loss.item():.4f}")

# ==========================================
# 4. API ENDPOINTS
# ==========================================
class InteractionRequest(BaseModel):
    drug_a_name: str
    drug_b_name: str

@app.get("/drugs")
def get_drugs():
    return {"drugs": sorted(list(drug_db.keys()))}

@app.post("/predict")
def predict_interaction(request: InteractionRequest):
    s1 = drug_db.get(request.drug_a_name)
    s2 = drug_db.get(request.drug_b_name)
    
    if not s1 or not s2:
        return {"error": "Drug not found"}
        
    v1 = smiles_to_vector(s1)
    v2 = smiles_to_vector(s2)
    
    combined = np.concatenate([v1, v2])
    tensor_in = torch.FloatTensor(combined)
    
    with torch.no_grad():
        prob = model(tensor_in).item()
        
    return {
        "drug_a": request.drug_a_name,
        "drug_b": request.drug_b_name,
        "probability": prob,
        "risk_level": "High" if prob > 0.5 else "Low"
    }
