# DataMine Studio — Complete Data Mining Dashboard

Educational full-stack Data Mining dashboard.

## Stack
- Frontend: React + Vite + TypeScript + Tailwind CSS
- Backend: Python + FastAPI + Pandas + Scikit-learn
- Algorithms: K-Means, K-Medoids, DBSCAN, AGNES, DIANA
- Evaluation: Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz
- PCA visualization and CSV/XLSX export

## Requirements
- Node.js 18+
- Python 3.10+

## Run backend
```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Run frontend
Open another terminal:
```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal.

## Workflow
1. Upload CSV/XLSX.
2. Review missing values and column types.
3. Select numeric features.
4. Apply missing-value handling and normalization.
5. Run a clustering algorithm.
6. Inspect metrics, PCA and cluster assignments.
7. Export the result.

## API
- `GET /api/health`
- `POST /api/upload`
- `POST /api/preprocess`
- `POST /api/cluster`
- `POST /api/evaluate`
- `POST /api/pca`
- `POST /api/export`
