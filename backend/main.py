from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import io, uuid
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

app = FastAPI(title="DataMine Studio API", version="1.0.0")
DATASETS = {}

class DataRequest(BaseModel):
    dataset_id: str
    features: list[str]
    missing_strategy: str = "median"
    scaler: str = "standard"

class ClusterRequest(DataRequest):
    algorithm: str
    k: int = 3
    eps: float = 0.5
    min_samples: int = 5
    linkage: str = "ward"

def read_file(content, filename):
    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(content))
        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")

def prepare(req):
    if req.dataset_id not in DATASETS:
        raise HTTPException(404, "Dataset not found")
    df = DATASETS[req.dataset_id].copy()
    missing = [c for c in req.features if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Unknown columns: {missing}")
    X = df[req.features].apply(pd.to_numeric, errors="coerce")
    if req.missing_strategy == "drop":
        X = X.dropna()
    elif req.missing_strategy == "mean":
        X = X.fillna(X.mean(numeric_only=True))
    elif req.missing_strategy == "mode":
        for c in X.columns:
            mode = X[c].mode()
            X[c] = X[c].fillna(mode.iloc[0] if len(mode) else 0)
    else:
        X = X.fillna(X.median(numeric_only=True))
    if X.isna().any().any():
        X = X.fillna(0)
    if req.scaler == "minmax":
        Xs = MinMaxScaler().fit_transform(X)
    elif req.scaler == "robust":
        Xs = RobustScaler().fit_transform(X)
    else:
        Xs = StandardScaler().fit_transform(X)
    return X, np.asarray(Xs)

def kmedoids(X, k, max_iter=100):
    rng = np.random.default_rng(42)
    n = len(X)
    if k > n:
        raise ValueError("k cannot exceed number of rows")
    medoids = rng.choice(n, k, replace=False)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    for _ in range(max_iter):
        labels = D[:, medoids].argmin(axis=1)
        new_medoids = medoids.copy()
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members):
                costs = D[np.ix_(members, members)].sum(axis=1)
                new_medoids[j] = members[costs.argmin()]
        if np.array_equal(new_medoids, medoids):
            break
        medoids = new_medoids
    labels = D[:, medoids].argmin(axis=1)
    return labels, medoids

def diana(X, k):
    # Simple divisive hierarchical clustering using maximum-distance splits.
    clusters = [np.arange(len(X))]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    while len(clusters) < min(k, len(X)):
        idx = int(np.argmax([D[np.ix_(c,c)].mean() if len(c)>1 else -1 for c in clusters]))
        c = clusters.pop(idx)
        if len(c) <= 1:
            clusters.append(c); break
        a, b = [c[np.argmax(D[np.ix_(c,c)].mean(axis=1))]]
        remaining = [x for x in c if x not in (a,b)]
        A, B = [a], [b]
        for x in remaining:
            if D[x, A].mean() <= D[x, B].mean(): A.append(x)
            else: B.append(x)
        clusters.extend([np.array(A), np.array(B)])
    labels = np.zeros(len(X), dtype=int)
    for i,c in enumerate(clusters):
        labels[c] = i
    return labels

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    df = read_file(content, file.filename)
    dataset_id = str(uuid.uuid4())
    DATASETS[dataset_id] = df
    info = {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "columns_info": [
            {"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()),
             "unique": int(df[c].nunique(dropna=True))}
            for c in df.columns
        ],
        "preview": df.head(10).replace({np.nan: None}).to_dict(orient="records")
    }
    return info

@app.post("/api/preprocess")
def preprocess(req: DataRequest):
    X, Xs = prepare(req)
    return {
        "rows": len(X),
        "features": req.features,
        "preview": pd.DataFrame(Xs, columns=req.features).head(10).round(4).to_dict(orient="records")
    }

@app.post("/api/cluster")
def cluster(req: ClusterRequest):
    X, Xs = prepare(req)
    algo = req.algorithm.lower()
    if algo == "kmeans":
        model = KMeans(n_clusters=req.k, random_state=42, n_init=10)
        labels = model.fit_predict(Xs)
        centers = model.cluster_centers_.round(5).tolist()
        extra = {"centers": centers}
    elif algo == "kmedoids":
        labels, medoids = kmedoids(Xs, req.k)
        extra = {"medoid_indices": medoids.tolist(), "medoids": Xs[medoids].round(5).tolist()}
    elif algo == "dbscan":
        model = DBSCAN(eps=req.eps, min_samples=req.min_samples)
        labels = model.fit_predict(Xs)
        extra = {"noise_points": int((labels == -1).sum())}
    elif algo == "agnes":
        if req.linkage == "ward":
            model = AgglomerativeClustering(n_clusters=req.k, linkage="ward")
        else:
            model = AgglomerativeClustering(n_clusters=req.k, linkage=req.linkage)
        labels = model.fit_predict(Xs)
        extra = {}
    elif algo == "diana":
        labels = diana(Xs, req.k)
        extra = {}
    else:
        raise HTTPException(400, "Unknown algorithm")

    result = pd.DataFrame(X, columns=req.features)
    result["cluster"] = labels
    metrics = {}
    valid = labels != -1
    unique = np.unique(labels[valid])
    if valid.sum() > 1 and len(unique) > 1:
        metrics = {
            "silhouette": round(float(silhouette_score(Xs[valid], labels[valid])), 4),
            "davies_bouldin": round(float(davies_bouldin_score(Xs[valid], labels[valid])), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(Xs[valid], labels[valid])), 4)
        }
    return {
        "algorithm": req.algorithm.upper(),
        "rows": len(result),
        "cluster_counts": result["cluster"].value_counts().sort_index().to_dict(),
        "labels": labels.tolist(),
        "metrics": metrics,
        **extra
    }

@app.post("/api/evaluate")
def evaluate(req: ClusterRequest):
    X, Xs = prepare(req)
    ks = list(range(2, min(10, len(X)-1) + 1))
    scores = []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(Xs)
        scores.append({
            "k": k,
            "inertia": round(float(model.inertia_), 4),
            "silhouette": round(float(silhouette_score(Xs, labels)), 4)
        })
    return {"scores": scores}

@app.post("/api/pca")
def pca(req: DataRequest):
    X, Xs = prepare(req)
    p = PCA(n_components=min(3, Xs.shape[1], Xs.shape[0]))
    coords = p.fit_transform(Xs)
    result = [{"PC1": round(float(r[0]), 5),
               "PC2": round(float(r[1]), 5) if coords.shape[1] > 1 else 0,
               "PC3": round(float(r[2]), 5) if coords.shape[1] > 2 else 0}
              for r in coords]
    return {"points": result, "explained_variance": [round(float(x), 5) for x in p.explained_variance_ratio_]}

@app.post("/api/export")
def export(req: ClusterRequest):
    X, Xs = prepare(req)
    algo = req.algorithm.lower()
    if algo == "kmeans":
        labels = KMeans(n_clusters=req.k, random_state=42, n_init=10).fit_predict(Xs)
    elif algo == "kmedoids":
        labels, _ = kmedoids(Xs, req.k)
    elif algo == "dbscan":
        labels = DBSCAN(eps=req.eps, min_samples=req.min_samples).fit_predict(Xs)
    elif algo == "agnes":
        labels = AgglomerativeClustering(n_clusters=req.k, linkage=req.linkage).fit_predict(Xs)
    else:
        labels = diana(Xs, req.k)
    out = X.copy()
    out["cluster"] = labels
    buf = io.BytesIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="clustering_results.csv"'})
