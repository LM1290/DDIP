import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from fastapi import FastAPI
from pydantic import BaseModel
import random
from real_data import get_real_dataset

app = FastAPI()

# --- 1. CONFIG & MODEL ARCHITECTURE ---
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


# --- 2. INITIALIZATION (Real Data) ---
print("Initializing BioGuard... Loading Curated Interaction Database...")

# Load real data from our helper script
drug_db, df_train = get_real_dataset()

print(f"Training on {len(df_train)} clinically validated pairs...")

# Prepare Tensors
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

# Train Model
model = DDI_Network()
# Use Weighted Loss if dataset is imbalanced, but BCELoss is fine for now
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Standard learning rate

model.train()
# Train longer because real chemistry is harder to learn than random noise
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

model.eval()
print(f"Model Converged. Final Loss: {loss.item():.4f}")


# --- 3. HELPER FUNCTIONS ---
def smiles_to_vector(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.zeros(N_BITS)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=N_BITS)
    arr = np.zeros((1,))
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# --- 4. API ENDPOINTS ---

class InteractionRequest(BaseModel):
    drug_a_name: str
    drug_b_name: str


@app.get("/drugs")
def get_drugs():
    """Returns list of available drugs for the iOS dropdown"""
    return {"drugs": list(drug_db.keys())}


@app.post("/predict")
def predict_interaction(request: InteractionRequest):
    # 1. Lookup SMILES
    s1 = drug_db.get(request.drug_a_name)
    s2 = drug_db.get(request.drug_b_name)

    if not s1 or not s2:
        return {"error": "Drug not found"}

    # 2. Vectorize
    v1 = smiles_to_vector(s1)
    v2 = smiles_to_vector(s2)

    # 3. Concatenate & Tensor
    combined = np.concatenate([v1, v2])
    tensor_in = torch.FloatTensor(combined)

    # 4. Predict
    with torch.no_grad():
        prob = model(tensor_in).item()

    return {
        "drug_a": request.drug_a_name,
        "drug_b": request.drug_b_name,
        "probability": prob,
        "risk_level": "High" if prob > 0.5 else "Low"
    }

# Run with: uvicorn backend:app --host 0.0.0.0 --port 8000
