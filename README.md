DISCLAIMER: Model was trained on a synthetic dataset and IS NOT intended for any practical clinical applications. 



![ddip](https://github.com/user-attachments/assets/acb3befc-ed3d-4cbb-8c89-0e0ace5abcf0)

BioGuard: Deep Learning DDI Predictor

A full-stack prototype for predicting adverse Drug-Drug Interactions (DDI) using chemical structure embeddings.

Architecture

This project uses a decoupled client–server architecture:
	•	Backend (Microservice): Python (FastAPI) with PyTorch
	•	RDKit generates Morgan Fingerprints (ECFP4) from SMILES strings
	•	A three-layer feed-forward neural network performs binary classification for interaction probability
	•	Packaged in Docker for cloud deployment (Render, Railway)
	•	Frontend (Client): Native iOS application written in SwiftUI
	•	Modern, pharmaceutical-oriented interface
	•	Asynchronous networking to avoid main-thread blocking

Tech Stack
	•	Machine Learning: PyTorch, RDKit, NumPy, Pandas
	•	API Layer: FastAPI, Uvicorn, Docker
	•	Mobile: Swift, SwiftUI, Combine

Methodology (Prototype)
	•	Data: Synthetic pipeline generating valid chemical structures (SMILES) mapped to drug entities
	•	Feature Engineering: 2048-bit Morgan Fingerprints (radius 2)
	•	Model: Binary classification MLP trained with binary cross-entropy loss and Adam optimization

Setup

Backend
cd backend
docker build -t ddi-backend .
docker run -p 8000:8000 ddi-backend
iOS App
	1.	Open ios-app/BioGuard.xcodeproj in Xcode.
	2.	Update the baseURL value in ContentView.swift to point to your local or deployed backend.
