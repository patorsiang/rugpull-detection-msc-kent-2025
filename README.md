# Rug-pull Scam Detector

This repository contains the **Rug-pull Scam Detection System**, developed as part of the MSc Advanced Computer Science dissertation at the University of Kent.

The project focuses on the **automatic detection and categorisation** of cryptocurrency rug-pull scams using blockchain transaction data, bytecode, and smart contract analysis.

This repository includes the source code, datasets, and usage instructions.

## 📦 1. Requirements

- **Python:** 3.12+
- **Node.js:** 18+
- **Docker:** 24+
- **Disk Space:** At least 16 GB
- **RAM:** Minimum 8 GB, 16 GB recommended

### Environment Variables

- Copy `.env.local` to `.env`

```env
ETHERSCAN_API_KEY=<your_etherscan_api_key>
REDIS_HOST=localhost
REDIS_PORT=6379
VITE_CURL_ENDPOINT="http://localhost:8000/api/predict"
```

**Notes:**

- Get your own API key from Etherscan (free account).
- The backend uses this key for blockchain data retrieval.

## ⚙️ 2. Installation

### **Option 1 – Using Docker (Recommended)**

```bash
# Clone the repository
git clone https://git.cs.kent.ac.uk/nt375/rugpull-detection-msc-kent-2025.git
cd rugpull-detection-msc-kent-2025

# Make sure that the submodules will be cloned together
git submodule update --init --force --remote

# Build and start containers
docker compose up --build -d
```

After a successful build:

- Backend API → <http://localhost:8000/docs>
- Frontend UI → <http://localhost:5173>

### Option 2 – Manual Setup (Without Docker)

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev --host 0.0.0.0
```

#### Radis

```bash
brew install redis
redis-server ./redis.conf
```

## ▶️ 3. Usage

### Using the Web Interface

1. Open <http://localhost:5173>
2. Paste contract addresses and click Predict to see results.
![paste contract addresses](/snapshots/1.png)
3. View predictions, label probabilities, and anomaly scores.
![paste contract addresses](/snapshots/7.png)
![paste contract addresses](/snapshots/8.png)

### Access API Documentation

Visit: <http://localhost:8000/docs>

You can run requests directly from the Swagger UI.

1. check health of the system

    ![check health](/snapshots/2.png)

    - /health → Shows whether the system has a trained model
    - /quota → Displays remaining Etherscan API quota
    - /versions → Shows meta information about the system

2. lookup the data in the system

    ![check feature](/snapshots/3.png)

    ![check dataset file in data folder](/snapshots/4.png)

    List of dataset files in the data folder.

    ![check feature of each address on selected file](/snapshots/5.png)

    **Note:** Setting `refresh` to `true` reloads transaction data. The address list is specified in the request body.

    ![set](/snapshots/6.png)

3. if you want train you own

    - add csv file to data folder
    - ![tuning on training](/snapshots/9.png)
      - `test_size` → Default is 0.2 (20% test, 80% train)
      - `N_TRIALS` → Number of Optuna hyperparameter tuning trials
      - `source` → Dataset file name in the data folder
      - Note: it's take time about 30 mins to 1 hours for 300 records
    - To run the best-tuned model on the full dataset:
      - ![finalized train](/snapshots/10.png)
      - This prepares the model for self-learning (pseudo-labeling on unlabeled data).
    - To run self-learning with tuning:
      - ![self learning and training more](/snapshots/11.png)
        - Pseudo-label and merge with ground truth (`source`)
        - eval_s  ource → Dataset for evaluation before merging
        - `N_TRIALS` → Number of trials for tuning
        - `preview_trials` → Preview trial results
        - `accept_min_delta` (optional) → Minimum accuracy/F1 improvement before merging
        - `chunk_size / chunk_index` (optional) → For large datasets
        - `low / high` → Confidence thresholds
        - Note: it's take time about 30 mins to 1 hours for 300 records

      - To pseudo-label files containing extra labels:
        - ![expand label](/snapshots/12.png)
        - `low / high` → Confidence thresholds
        - `source` → Dataset file name in the data folder
        - `chunk_size / chunk_index` (optional) → For large datasets
        - `filtered`: want to filter -1 or not

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

**Note:** Now using groundtruth_v3.csv based
