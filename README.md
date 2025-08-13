# Rugpull Scam Detector

This repository contains the Rugpull Scam Detection System, developed as part of the MSc Advanced Computer Science dissertation at the University of Kent.
The project focuses on automatic detection and categorisation of cryptocurrency rug pull scams using blockchain transaction data, bytecode, and smart contract analysis.

## Structure

```txt
backend/                 # FastAPI backend and ML models
  └── models/current/     # Current deployed model files
data/
  ├── features/           # Extracted features for ML models
  ├── hex/                # EVM bytecode (.hex) files
  ├── sol/                # Solidity source code (.sol) files
  ├── txn/                # Transaction and event log data
  └── *.csv               # labels
```

## Features

- Multi-source feature extraction:
  - Opcode sequences & byte frequency
  - Transaction-level statistics
  - Graph-based contract analysis
- Multi-label classification (e.g., Mint, Leak, Limit)
- Self-learning & incremental learning for continuous improvement
- Anomaly detection for identifying suspicious behaviors
- FastAPI backend for API-based predictions
- Dockerized environment for easy deployment

## Running Locally

### Prerequisites

--> Docker and Docker Compose

### Start the Services

```sh
docker-compose up --build -d
```

### Access the API

Swagger Docs (Training & Prediction APIs): <http://localhost:8000/docs>

Frontend (Final Demo UI): <http://localhost:5173>

## Related Docs

- [planning](/PLANNING.md)

**note:** This repository is for academic purposes only.
