import React, { useState, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import {
  Upload, Database, SlidersHorizontal, BrainCircuit, BarChart3, Download, CheckCircle2, Play,
  FileSpreadsheet, BookOpen, Lightbulb, ArrowRight, Info, Sparkles, RotateCcw, Shuffle, Check,
} from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell,
  ScatterChart, Scatter, Legend,
} from "recharts";
import "./index.css";
import "./theme.css";

const API = `${import.meta.env.VITE_API_URL}/api`;
const NUMERIC = ["int64", "float64", "int32", "float32"];
const COLORS = ["#1e3a5f", "#6b8e4e", "#b08d4a", "#9b2c2c", "#6d4c7d", "#2f7f86", "#c26a4a", "#7a7f3a", "#8a6a3f", "#4a5568"];
const NOISE_COLOR = "#a8a29e";

type Info = { dataset_id: string; filename: string; rows: number; columns: number; columns_info: any[]; preview: any[] };
type Result = { algorithm: string; metrics: any; cluster_counts: any; labels: number[]; centers?: any; medoids?: any };
type Run = { id: number; algorithm: string; k?: number; eps?: number; minSamples?: number; silhouette: number | null; clusters: number; scaler: string; strategy: string };

/* ----------------------------- DWM knowledge base ----------------------------- */

const ALGOS: Record<string, any> = {
  kmeans: {
    name: "K-Means", type: "Partitioning",
    desc: "Splits data into K groups by repeatedly assigning points to the nearest centroid and moving each centroid to the mean of its group.",
    pros: ["Fast and scalable", "Simple to interpret"], cons: ["You must choose K", "Sensitive to outliers", "Prefers round clusters"],
    use: "Numeric data, roughly spherical clusters, K is known or can be estimated with the elbow method.",
  },
  kmedoids: {
    name: "K-Medoids (PAM)", type: "Partitioning",
    desc: "Like K-Means, but each cluster is represented by a real data point (the medoid) instead of a mean.",
    pros: ["Robust to outliers", "Centers are real records"], cons: ["Slower on big data", "You must choose K"],
    use: "Data with outliers or when you need an actual representative record per cluster.",
  },
  dbscan: {
    name: "DBSCAN", type: "Density-based",
    desc: "Grows clusters from dense regions: a point with at least MinPts neighbours within radius Eps is a core point. Sparse points become noise.",
    pros: ["Finds any shape", "Detects noise/outliers", "No K needed"], cons: ["Eps/MinPts are tricky", "Struggles with varying density"],
    use: "Irregular cluster shapes or data where noise detection matters.",
  },
  agnes: {
    name: "AGNES", type: "Hierarchical (bottom-up)",
    desc: "Agglomerative: starts with every point as its own cluster and repeatedly merges the two closest clusters (Ward linkage here).",
    pros: ["Gives a full hierarchy", "Deterministic"], cons: ["O(n²) memory", "Merges can't be undone"],
    use: "Small/medium datasets where you want to explore structure at several levels.",
  },
  diana: {
    name: "DIANA", type: "Hierarchical (top-down)",
    desc: "Divisive: starts with one big cluster and recursively splits the most heterogeneous cluster.",
    pros: ["Good at finding big clusters first", "Hierarchy view"], cons: ["Computationally heavy", "Splits can't be undone"],
    use: "When the major groups matter more than fine detail.",
  },
};

const MISSING: Record<string, [string, string]> = {
  median: ["Median", "Fills gaps with the middle value. Robust against outliers, so it's a safe default."],
  mean: ["Mean", "Fills gaps with the average. Good for symmetric data without extreme values."],
  mode: ["Mode", "Fills gaps with the most frequent value. Best for categorical-like numeric codes."],
  drop: ["Drop rows", "Removes incomplete records. Clean, but you lose data if many values are missing."],
};
const SCALERS: Record<string, [string, string, string]> = {
  standard: ["Z-score (Standard)", "(x − mean) / std", "Centers data at 0 with unit variance. Best general choice for K-Means and PCA."],
  minmax: ["Min-Max", "(x − min) / (max − min)", "Squeezes every feature into [0, 1]. Keeps shape, but outliers squash everything else."],
  robust: ["Robust", "(x − median) / IQR", "Uses median and quartiles, so outliers barely affect the scaling."],
};
const GLOSS: Record<string, string> = {
  silhouette: "How well each point fits its own cluster versus the nearest other cluster. Ranges −1 to 1; higher is better.",
  davies_bouldin: "Average similarity between each cluster and its most similar one. Lower is better (0 is ideal).",
  calinski_harabasz: "Ratio of between-cluster to within-cluster spread. Higher means denser, better-separated clusters.",
  inertia: "Sum of squared distances from points to their centroid (SSE). Lower is tighter, but always falls as K grows.",
  n_clusters: "Number of clusters found.",
  noise_points: "Points DBSCAN labelled as noise (outliers) that belong to no cluster.",
};
const CONCEPTS: [string, string][] = [
  ["KDD process", "Knowledge Discovery in Databases: Selection → Preprocessing → Transformation → Data Mining → Interpretation. This app follows exactly this pipeline."],
  ["Data cleaning", "Handling missing values, noise and inconsistencies. Strategies here: mean/median/mode imputation or dropping rows."],
  ["Data transformation", "Normalization rescales attributes so distance-based algorithms don't let large-range features dominate."],
  ["Data reduction (PCA)", "Principal Component Analysis projects many features onto a few components that keep most of the variance, making clusters visible in 2D."],
  ["Proximity measure", "Clustering needs a notion of 'similar'. Euclidean distance is the default for numeric attributes."],
  ["Partitioning methods", "K-Means and K-Medoids divide n objects into K flat clusters, optimizing an objective such as SSE."],
  ["Hierarchical methods", "AGNES (bottom-up) and DIANA (top-down) build a tree of clusters (dendrogram)."],
  ["Density-based methods", "DBSCAN defines clusters as dense regions separated by sparse ones, and flags outliers as noise."],
  ["Cluster validity", "Internal indices (Silhouette, Davies-Bouldin, Calinski-Harabasz) judge cluster quality without labels. The elbow method plots SSE against K."],
  ["Outlier analysis", "Outliers distort means (K-Means) but are naturally isolated by DBSCAN and tolerated by K-Medoids."],
];

/* ----------------------------- helpers ----------------------------- */

const silhouetteLabel = (s: any): [string, string] => {
  if (typeof s !== "number") return ["Not available", "text-[#5b6478]"];
  if (s > 0.7) return ["Strong structure", "text-green-600"];
  if (s > 0.5) return ["Reasonable structure", "text-emerald-600"];
  if (s > 0.25) return ["Weak structure", "text-amber-600"];
  return ["Little or no structure", "text-red-600"];
};
const prettyNum = (v: any) => (typeof v === "number" ? (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3)) : String(v));
const autoFeatures = (cols: any[]) =>
  cols.filter((x) => NUMERIC.includes(x.dtype) && !/(^id$|_id$|^id_|index)/i.test(x.name)).map((x) => x.name);

function makeSampleCsv() {
  const n = (m: number, s: number) => m + (Math.random() + Math.random() + Math.random() - 1.5) * s;
  const groups = [
    { a: 90, h: 14, i: 82, s: 85, e: 88 },
    { a: 75, h: 8, i: 64, s: 68, e: 66 },
    { a: 55, h: 3, i: 42, s: 45, e: 38 },
  ];
  const rows = ["student_id,attendance,study_hours,internal_marks,assignment_score,exam_score"];
  for (let i = 1; i <= 150; i++) {
    const g = groups[i % 3];
    const v = [n(g.a, 8), n(g.h, 3), n(g.i, 9), n(g.s, 9), n(g.e, 9)].map((x) => Math.max(0, +x.toFixed(1)));
    const cells = v.map((x) => (Math.random() < 0.03 ? "" : x));
    rows.push([i, ...cells].join(","));
  }
  return rows.join("\n");
}

/* ----------------------------- App ----------------------------- */

const TABS: [string, any][] = [
  ["Dashboard", Database], ["Upload", Upload], ["Preprocess", SlidersHorizontal], ["Clustering", BrainCircuit],
  ["Evaluation", BarChart3], ["Results", CheckCircle2], ["Learn", BookOpen],
];

function App() {
  const [tab, setTab] = useState("Dashboard");
  const [visited, setVisited] = useState<Set<string>>(new Set(["Dashboard"]));
  const [info, setInfo] = useState<Info | null>(null);
  const [features, setFeatures] = useState<string[]>([]);
  const [strategy, setStrategy] = useState("median");
  const [scaler, setScaler] = useState("standard");
  const [algorithm, setAlgorithm] = useState("kmeans");
  const [k, setK] = useState(3);
  const [eps, setEps] = useState(0.5);
  const [minSamples, setMinSamples] = useState(5);
  const [result, setResult] = useState<Result | null>(null);
  const [evalData, setEvalData] = useState<any[]>([]);
  const [pca, setPca] = useState<any | null>(null);
  const [history, setHistory] = useState<Run[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");

  const go = (t: string) => { setTab(t); setVisited((v) => new Set(v).add(t)); window.scrollTo({ top: 0, behavior: "smooth" }); };
  const notify = (m: string) => { setToast(m); setTimeout(() => setToast(""), 2800); };

  async function upload(file: File) {
    setBusy(true); setError("");
    const fd = new FormData(); fd.append("file", file);
    try {
      const r = await fetch(API + "/upload", { method: "POST", body: fd });
      if (!r.ok) throw Error(await r.text());
      const d = await r.json();
      setInfo(d); setFeatures(autoFeatures(d.columns_info));
      setResult(null); setEvalData([]); setPca(null);
      notify(`Loaded ${d.filename}: ${d.rows} rows × ${d.columns} columns`);
      go("Preprocess");
    } catch (e: any) { setError(e.message || "Upload failed"); }
    finally { setBusy(false); }
  }
  const loadSample = () => upload(new File([makeSampleCsv()], "student_performance_sample.csv", { type: "text/csv" }));

  const payload = (o: any = {}) => ({
    dataset_id: info?.dataset_id, features, missing_strategy: strategy, scaler, k, eps,
    min_samples: minSamples, algorithm, linkage: "ward", ...o,
  });
  const post = (path: string, body: any) =>
    fetch(API + path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  async function run(o: any = {}) {
    if (!info) return setError("Upload a dataset first.");
    if (features.length < 2) return setError("Select at least 2 numeric features in the Preprocess step.");
    setBusy(true); setError("");
    const body = payload(o);
    try {
      const r = await post("/cluster", body);
      if (!r.ok) throw Error(await r.text());
      const res: Result = await r.json();
      setResult(res);
      const e = await post("/evaluate", body);
      if (e.ok) setEvalData((await e.json()).scores || []);
      const p = await post("/pca", body);
      if (p.ok) setPca(await p.json());
      setHistory((h) => [{
        id: h.length + 1, algorithm: body.algorithm, k: body.algorithm === "dbscan" ? undefined : body.k,
        eps: body.algorithm === "dbscan" ? body.eps : undefined, minSamples: body.algorithm === "dbscan" ? body.min_samples : undefined,
        silhouette: typeof res.metrics?.silhouette === "number" ? res.metrics.silhouette : null,
        clusters: Object.keys(res.cluster_counts || {}).length, scaler, strategy,
      }, ...h]);
      notify("Clustering complete");
      go("Results");
    } catch (er: any) { setError(er.message || "Clustering failed"); }
    finally { setBusy(false); }
  }

  async function exportResult() {
    if (!info) return;
    const r = await post("/export", payload());
    const b = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b); a.download = "clustering_results.csv"; a.click();
  }

  const stages = [
    { label: "Selection", tab: "Upload", done: !!info },
    { label: "Preprocessing", tab: "Preprocess", done: !!info && visited.has("Preprocess") },
    { label: "Data Mining", tab: "Clustering", done: !!result },
    { label: "Interpretation", tab: "Results", done: !!result && visited.has("Results") },
  ];

  return (
    <div className="min-h-screen flex bg-[#f4efe6]">
      <aside className="w-64 bg-[#fffdf8] border-r border-[#ddd3c0] p-5 hidden md:block sticky top-0 h-screen">
        <div className="flex items-center gap-2 mb-8">
          <div className="bg-[#1e2a44] text-white p-2 rounded-xl"><BrainCircuit size={20} /></div>
          <div><b>DataMine</b><p className="text-xs text-[#8b93a5]">Discover hidden patterns in data.</p></div>
        </div>
        <div className="space-y-1">
          {TABS.map(([n, I]) => (
            <button key={n} onClick={() => go(n)} className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition ${tab === n ? "bg-[#1e2a44] text-white" : "text-[#4a5368] hover:bg-[#efe8d8]"}`}>
              <I size={17} />{n}
            </button>
          ))}
        </div>
        <div className="mt-8 p-4 rounded-xl bg-[#f8f3e8] text-xs text-[#5b6478]">
          Data Warehousing &amp; Mining Lab<br /><span className="text-gray-800 font-semibold">5 clustering algorithms · KDD pipeline</span>
        </div>
      </aside>

      <main className="flex-1 p-4 md:p-8 max-w-[1500px] mx-auto w-full min-w-0">
        <div className="md:hidden flex gap-2 overflow-x-auto pb-3 mb-3">
          {TABS.map(([n]) => (
            <button key={n} onClick={() => go(n)} className={`px-3 py-1.5 rounded-full text-sm whitespace-nowrap ${tab === n ? "bg-[#1e2a44] text-white" : "bg-[#fffdf8] border text-[#4a5368]"}`}>{n}</button>
          ))}
        </div>

        <header className="flex justify-between items-center mb-5">
          <div>
            <h1 className="text-2xl font-bold text-[#1e2a44]">{tab}</h1>
            <p className="text-sm text-[#5b6478]">{HEADLINES[tab]}</p>
          </div>
          {info && <div className="hidden sm:flex items-center gap-2 text-sm bg-[#fffdf8] border px-3 py-2 rounded-xl"><FileSpreadsheet size={16} />{info.filename}</div>}
        </header>

        <KddStepper stages={stages} go={go} />

        {error && (
          <div className="mb-5 p-3 rounded-xl bg-red-50 text-red-700 text-sm flex justify-between gap-3">
            <span className="break-words">{error}</span>
            <button className="font-bold" onClick={() => setError("")}>✕</button>
          </div>
        )}

        {tab === "Dashboard" && <Dashboard info={info} result={result} history={history} onUpload={upload} loadSample={loadSample} busy={busy} go={go} />}
        {tab === "Upload" && <UploadPage onUpload={upload} loadSample={loadSample} busy={busy} info={info} go={go} />}
        {tab === "Preprocess" && <Preprocess info={info} features={features} setFeatures={setFeatures} strategy={strategy} setStrategy={setStrategy} scaler={scaler} setScaler={setScaler} go={go} />}
        {tab === "Clustering" && <ClusterPage info={info} algorithm={algorithm} setAlgorithm={setAlgorithm} k={k} setK={setK} eps={eps} setEps={setEps} minSamples={minSamples} setMinSamples={setMinSamples} run={() => run()} busy={busy} features={features} />}
        {tab === "Evaluation" && <Evaluation data={evalData} result={result} k={k} algorithm={algorithm} useBestK={(bk: number) => { setK(bk); run({ k: bk }); }} go={go} />}
        {tab === "Results" && <Results result={result} pca={pca} history={history} exportResult={exportResult} go={go} />}
        {tab === "Learn" && <Learn />}
      </main>

      {toast && <div className="fixed bottom-5 right-5 bg-[#1e2a44] text-white text-sm px-4 py-3 rounded-xl shadow-lg flex items-center gap-2"><Check size={16} />{toast}</div>}
    </div>
  );
}

const HEADLINES: Record<string, string> = {
  Dashboard: "Your data mining workspace at a glance.",
  Upload: "Step 1 · Selection — choose the data to mine.",
  Preprocess: "Step 2 · Preprocessing — clean, select and transform.",
  Clustering: "Step 3 · Data mining — pick an algorithm and tune it.",
  Evaluation: "Step 4 · Judge cluster quality with validity indices.",
  Results: "Step 5 · Interpretation — understand and export the clusters.",
  Learn: "DWM concepts and a hands-on K-Means playground.",
};

/* ----------------------------- shared UI ----------------------------- */

function KddStepper({ stages, go }: any) {
  return (
    <div className="card p-4 mb-6 overflow-x-auto">
      <div className="flex items-center min-w-[520px]">
        {stages.map((s: any, i: number) => (
          <React.Fragment key={s.label}>
            <button onClick={() => go(s.tab)} className="flex items-center gap-2 group">
              <span className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${s.done ? "bg-green-600 text-white" : "bg-[#efe8d8] text-[#5b6478] group-hover:bg-gray-200"}`}>
                {s.done ? <Check size={14} /> : i + 1}
              </span>
              <span className={`text-sm ${s.done ? "text-[#1e2a44] font-semibold" : "text-[#5b6478]"}`}>{s.label}</span>
            </button>
            {i < stages.length - 1 && <div className={`flex-1 h-0.5 mx-3 ${stages[i].done ? "bg-green-600" : "bg-gray-200"}`} />}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

function Tip({ text }: { text: string }) {
  return <span title={text} className="inline-flex align-middle ml-1 text-[#8b93a5] cursor-help"><Info size={14} /></span>;
}
function NextBar({ label, onClick, disabled }: any) {
  return (
    <div className="flex justify-end">
      <button className="btn btn-primary" onClick={onClick} disabled={disabled}>{label}<ArrowRight size={16} className="inline ml-2" /></button>
    </div>
  );
}
function Empty({ text, action }: any) {
  return <div className="card p-10 text-center text-[#5b6478]">{text}{action && <div className="mt-4">{action}</div>}</div>;
}
function Slider({ label, value, min, max, step, onChange, hint }: any) {
  return (
    <label className="block text-sm mt-5">
      <div className="flex justify-between"><span>{label}</span><b>{value}</b></div>
      <input type="range" className="w-full mt-2" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
      {hint && <p className="text-xs text-[#8b93a5] mt-1">{hint}</p>}
    </label>
  );
}

/* ----------------------------- pages ----------------------------- */

function Dashboard({ info, result, history, onUpload, loadSample, busy, go }: any) {
  const cols = info?.columns_info || [];
  const missing = cols.reduce((a: number, x: any) => a + (x.missing || 0), 0);
  const cells = info ? info.rows * info.columns : 0;
  const completeness = cells ? Math.round((1 - missing / cells) * 100) : null;
  const numeric = cols.filter((x: any) => NUMERIC.includes(x.dtype)).length;
  const best = history.filter((h: Run) => h.silhouette !== null).sort((a: Run, b: Run) => (b.silhouette as number) - (a.silhouette as number))[0];
  const next = !info ? ["Upload a dataset or try the sample", "Upload"] : !result ? ["Preprocess and run a clustering algorithm", "Clustering"] : ["Review your results and compare algorithms", "Results"];

  return (
    <div className="space-y-6">
      <div className="card p-7">
        <div className="max-w-2xl">
          <span className="text-xs font-bold uppercase tracking-widest text-[#8b93a5]">Data Mining Laboratory</span>
          <h2 className="text-4xl font-bold mt-2">Turn raw data into<br /><span className="text-[#5b6478]">discoverable patterns.</span></h2>
          <p className="text-[#5b6478] mt-4">Follow the KDD process end to end: select data, clean and transform it, mine it with five clustering algorithms, then evaluate and interpret the clusters.</p>
          <div className="flex flex-wrap gap-3 mt-6">
            <button className="btn btn-primary" onClick={() => document.getElementById("dashfile")?.click()}><Upload size={16} className="inline mr-2" />Upload Dataset</button>
            <button className="btn" onClick={loadSample} disabled={busy}><Sparkles size={16} className="inline mr-2" />{busy ? "Loading..." : "Try sample student data"}</button>
          </div>
          <input id="dashfile" type="file" accept=".csv,.xlsx,.xls" hidden onChange={(e) => e.target.files && onUpload(e.target.files[0])} />
        </div>
      </div>

      <button onClick={() => go(next[1])} className="w-full card p-4 flex items-center justify-between text-left hover:shadow-md transition">
        <span className="flex items-center gap-3"><Lightbulb className="text-amber-500" size={20} /><span><b>Suggested next step</b><span className="block text-sm text-[#5b6478]">{next[0]}</span></span></span>
        <ArrowRight size={18} />
      </button>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[["Rows", info?.rows || 0], ["Columns", info?.columns || 0], ["Numeric features", numeric], ["Data completeness", completeness === null ? "–" : completeness + "%"]].map((x) => (
          <div className="card p-5" key={x[0] as string}><p className="text-sm text-[#8b93a5]">{x[0]}</p><b className="text-3xl">{x[1]}</b></div>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-5">
        <div className="card p-6">
          <h3 className="font-bold mb-4">KDD pipeline</h3>
          {[["Selection", "Choose the target dataset", "Upload"], ["Preprocessing", "Clean missing values, pick features", "Preprocess"], ["Transformation", "Normalize attributes", "Preprocess"], ["Data mining", "Apply clustering", "Clustering"], ["Interpretation", "Evaluate and export", "Results"]].map((x, i) => (
            <button key={x[0]} onClick={() => go(x[2])} className="w-full flex items-center gap-3 py-2 text-left hover:bg-[#f8f3e8] rounded-lg px-2">
              <span className="w-7 h-7 rounded-full bg-[#efe8d8] flex items-center justify-center text-xs">{i + 1}</span>
              <span><b className="text-sm">{x[0]}</b><span className="block text-xs text-[#5b6478]">{x[1]}</span></span>
            </button>
          ))}
        </div>
        <div className="card p-6">
          <h3 className="font-bold mb-4">Latest model</h3>
          {result ? (
            <>
              <p className="text-2xl font-bold">{ALGOS[result.algorithm]?.name || result.algorithm}</p>
              <p className="text-[#5b6478] mt-2">Silhouette: <b>{prettyNum(result.metrics?.silhouette ?? "N/A")}</b> · <span className={silhouetteLabel(result.metrics?.silhouette)[1]}>{silhouetteLabel(result.metrics?.silhouette)[0]}</span></p>
              {best && <p className="text-sm text-[#5b6478] mt-3">Best run so far: <b>{ALGOS[best.algorithm]?.name}</b> (silhouette {prettyNum(best.silhouette)})</p>}
            </>
          ) : <p className="text-[#8b93a5]">Run a clustering algorithm to see results.</p>}
        </div>
      </div>
    </div>
  );
}

function UploadPage({ onUpload, loadSample, busy, info, go }: any) {
  const [drag, setDrag] = useState(false);
  return (
    <div className="space-y-5">
      <div className="card p-8">
        <div
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); e.dataTransfer.files[0] && onUpload(e.dataTransfer.files[0]); }}
          className={`border-2 border-dashed rounded-2xl p-16 text-center transition ${drag ? "border-[#1e2a44] bg-[#f8f3e8]" : "border-[#c9bda5]"}`}
        >
          <Upload className="mx-auto mb-4 text-[#8b93a5]" size={38} />
          <h2 className="text-xl font-bold">{drag ? "Release to upload" : "Drag & drop a CSV or Excel file"}</h2>
          <p className="text-[#5b6478] text-sm mt-2">Supported: .csv, .xlsx, .xls</p>
          <div className="flex flex-wrap gap-3 justify-center mt-5">
            <label className="btn btn-primary inline-block cursor-pointer">{busy ? "Uploading..." : "Choose file"}
              <input type="file" hidden accept=".csv,.xlsx,.xls" onChange={(e) => e.target.files && onUpload(e.target.files[0])} />
            </label>
            <button className="btn" onClick={loadSample} disabled={busy}><Sparkles size={16} className="inline mr-2" />Use sample data</button>
          </div>
        </div>
        <p className="text-xs text-[#8b93a5] mt-4">Tip: the sample dataset has 150 students with attendance, study hours and marks, and a few missing values to practise cleaning.</p>
      </div>

      {info && (
        <div className="card p-6">
          <div className="flex justify-between items-center mb-4">
            <div><b>{info.filename}</b><p className="text-sm text-[#5b6478]">{info.rows} rows × {info.columns} columns</p></div>
            <button className="btn btn-primary" onClick={() => go("Preprocess")}>Continue<ArrowRight size={16} className="inline ml-2" /></button>
          </div>
          <PreviewTable info={info} />
        </div>
      )}
    </div>
  );
}

function PreviewTable({ info }: any) {
  const names = info.columns_info.map((c: any) => c.name);
  return (
    <div className="overflow-auto max-h-72 border rounded-xl">
      <table className="text-xs w-full">
        <thead className="bg-[#f8f3e8] sticky top-0"><tr>{names.map((n: string) => <th key={n} className="text-left px-3 py-2 whitespace-nowrap">{n}</th>)}</tr></thead>
        <tbody>
          {(info.preview || []).slice(0, 15).map((r: any, i: number) => (
            <tr key={i} className="border-t">
              {names.map((n: string) => (
                <td key={n} className={`px-3 py-1.5 whitespace-nowrap ${r[n] === null || r[n] === undefined || r[n] === "" ? "bg-red-50 text-red-400" : ""}`}>
                  {r[n] === null || r[n] === undefined || r[n] === "" ? "missing" : String(r[n])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Preprocess({ info, features, setFeatures, strategy, setStrategy, scaler, setScaler, go }: any) {
  if (!info) return <Empty text="Upload a dataset first." action={<button className="btn btn-primary" onClick={() => go("Upload")}>Go to Upload</button>} />;
  const numericCols = info.columns_info.filter((c: any) => NUMERIC.includes(c.dtype));
  const toggle = (n: string) => setFeatures((f: string[]) => (f.includes(n) ? f.filter((x) => x !== n) : [...f, n]));
  const totalMissing = info.columns_info.reduce((a: number, c: any) => a + (c.missing || 0), 0);

  return (
    <div className="space-y-5">
      <div className="card p-6">
        <div className="flex flex-wrap justify-between gap-3 mb-1">
          <div>
            <h2 className="font-bold text-lg">1. Data profile &amp; feature selection<Tip text="Data reduction: dropping irrelevant attributes (like IDs) improves clustering quality." /></h2>
            <p className="text-sm text-[#5b6478]">Pick the numeric columns to mine. ID-like columns were excluded automatically. <b>{features.length}</b> of {numericCols.length} selected.</p>
          </div>
          <div className="flex gap-2 items-start">
            <button className="btn text-xs" onClick={() => setFeatures(numericCols.map((c: any) => c.name))}>Select all</button>
            <button className="btn text-xs" onClick={() => setFeatures(autoFeatures(info.columns_info))}>Smart select</button>
            <button className="btn text-xs" onClick={() => setFeatures([])}>Clear</button>
          </div>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-2 mt-4">
          {info.columns_info.map((c: any) => {
            const isNum = NUMERIC.includes(c.dtype);
            const pct = info.rows ? Math.round(((c.missing || 0) / info.rows) * 100) : 0;
            const on = features.includes(c.name);
            return (
              <label key={c.name} className={`border rounded-xl p-3 flex gap-3 items-center transition ${on ? "border-[#1e2a44] bg-[#f8f3e8]" : ""} ${isNum ? "cursor-pointer" : "opacity-50"}`}>
                <input type="checkbox" checked={on} disabled={!isNum} onChange={() => toggle(c.name)} />
                <span className="text-sm flex-1 min-w-0">
                  <span className="block truncate">{c.name}</span>
                  <small className="block text-[#8b93a5]">{c.dtype}{isNum ? "" : " · not numeric"}</small>
                  <span className="block h-1.5 bg-[#efe8d8] rounded mt-1"><span className={`block h-1.5 rounded ${pct > 20 ? "bg-red-500" : pct > 0 ? "bg-amber-400" : "bg-green-500"}`} style={{ width: `${Math.max(pct, 3)}%` }} /></span>
                  <small className="text-[#8b93a5]">{c.missing || 0} missing ({pct}%)</small>
                </span>
              </label>
            );
          })}
        </div>
        {features.length < 2 && <p className="text-sm text-amber-600 mt-4">Select at least 2 features to cluster.</p>}
      </div>

      <div className="grid md:grid-cols-2 gap-5">
        <div className="card p-6">
          <h3 className="font-bold">2. Data cleaning: missing values<Tip text="Imputation replaces missing entries with a statistic of the column." /></h3>
          <p className="text-sm text-[#5b6478] mb-4">{totalMissing ? `${totalMissing} missing values detected.` : "No missing values found. Any option works."}</p>
          <div className="space-y-2">
            {Object.entries(MISSING).map(([key, [name, desc]]) => (
              <button key={key} onClick={() => setStrategy(key)} className={`w-full text-left border rounded-xl p-3 transition ${strategy === key ? "border-[#1e2a44] bg-[#f8f3e8]" : "hover:bg-[#f8f3e8]"}`}>
                <b className="text-sm">{name}{key === "median" && <span className="ml-2 text-xs text-green-600">recommended</span>}</b>
                <span className="block text-xs text-[#5b6478] mt-0.5">{desc}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="card p-6">
          <h3 className="font-bold">3. Data transformation: normalization<Tip text="Without scaling, a feature measured in thousands dominates one measured in decimals." /></h3>
          <p className="text-sm text-[#5b6478] mb-4">Distance-based algorithms need features on comparable scales.</p>
          <div className="space-y-2">
            {Object.entries(SCALERS).map(([key, [name, formula, desc]]) => (
              <button key={key} onClick={() => setScaler(key)} className={`w-full text-left border rounded-xl p-3 transition ${scaler === key ? "border-[#1e2a44] bg-[#f8f3e8]" : "hover:bg-[#f8f3e8]"}`}>
                <b className="text-sm">{name}</b><code className="ml-2 text-xs bg-[#efe8d8] px-1.5 py-0.5 rounded">{formula}</code>
                <span className="block text-xs text-[#5b6478] mt-0.5">{desc}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="card p-6">
        <h3 className="font-bold mb-3">Data preview <span className="text-xs font-normal text-[#8b93a5]">(missing cells are highlighted)</span></h3>
        <PreviewTable info={info} />
      </div>
      <NextBar label="Continue to clustering" onClick={() => go("Clustering")} disabled={features.length < 2} />
    </div>
  );
}

function Recommender({ onPick }: { onPick: (a: string) => void }) {
  const [noise, setNoise] = useState<boolean | null>(null);
  const [hier, setHier] = useState<boolean | null>(null);
  const [outl, setOutl] = useState<boolean | null>(null);
  const rec = hier ? "agnes" : noise ? "dbscan" : outl ? "kmedoids" : "kmeans";
  const Q = ({ q, v, set }: any) => (
    <div className="flex items-center justify-between gap-3 py-1.5">
      <span className="text-sm">{q}</span>
      <span className="flex gap-1">
        {[true, false].map((b) => (
          <button key={String(b)} onClick={() => set(b)} className={`px-3 py-1 rounded-lg text-xs border ${v === b ? "bg-[#1e2a44] text-white" : "bg-white"}`}>{b ? "Yes" : "No"}</button>
        ))}
      </span>
    </div>
  );
  return (
    <div className="card p-6">
      <h3 className="font-bold mb-2"><Lightbulb size={16} className="inline mr-2 text-amber-500" />Not sure which algorithm?</h3>
      <Q q="Do you need a hierarchy (tree) of clusters?" v={hier} set={setHier} />
      <Q q="Do you expect odd shapes or noise points?" v={noise} set={setNoise} />
      <Q q="Does your data contain strong outliers?" v={outl} set={setOutl} />
      {(hier !== null || noise !== null || outl !== null) && (
        <div className="mt-3 p-3 rounded-xl bg-amber-50 text-sm flex justify-between items-center gap-3">
          <span>Recommended: <b>{ALGOS[rec].name}</b></span>
          <button className="btn text-xs" onClick={() => onPick(rec)}>Use it</button>
        </div>
      )}
    </div>
  );
}

function ClusterPage(p: any) {
  if (!p.info) return <Empty text="Upload a dataset first." />;
  const a = ALGOS[p.algorithm];
  return (
    <div className="grid lg:grid-cols-3 gap-5">
      <div className="space-y-5 lg:col-span-1">
        <div className="card p-6">
          <h2 className="font-bold text-lg">Algorithm &amp; parameters</h2>
          <select className="w-full mt-4" value={p.algorithm} onChange={(e) => p.setAlgorithm(e.target.value)}>
            {Object.entries(ALGOS).map(([key, v]: any) => <option key={key} value={key}>{v.name}</option>)}
          </select>
          {p.algorithm === "dbscan" ? (
            <>
              <Slider label="Epsilon (radius)" value={p.eps} min={0.1} max={3} step={0.1} onChange={p.setEps} hint="Larger radius merges clusters; smaller creates more noise." />
              <Slider label="Min samples" value={p.minSamples} min={2} max={20} step={1} onChange={p.setMinSamples} hint="Neighbours required to form a dense core point." />
            </>
          ) : (
            <Slider label="Number of clusters (K)" value={p.k} min={2} max={10} step={1} onChange={p.setK} hint="Not sure? Run once, then use the Evaluation tab to find the best K." />
          )}
          <p className="text-xs text-[#8b93a5] mt-4">Using {p.features.length} features.</p>
          <button className="btn btn-primary w-full mt-4" onClick={p.run} disabled={p.busy}>
            <Play size={16} className="inline mr-2" />{p.busy ? "Running..." : "Run algorithm"}
          </button>
        </div>
        <Recommender onPick={p.setAlgorithm} />
      </div>

      <div className="space-y-5 lg:col-span-2">
        <div className="card p-6">
          <div className="flex justify-between items-start">
            <div><h2 className="font-bold text-xl">{a.name}</h2><span className="text-xs bg-[#efe8d8] px-2 py-1 rounded-full">{a.type}</span></div>
          </div>
          <p className="text-[#4a5368] mt-4 text-sm">{a.desc}</p>
          <p className="text-sm mt-3"><b>Best for:</b> <span className="text-[#4a5368]">{a.use}</span></p>
          <div className="grid sm:grid-cols-2 gap-4 mt-4">
            <div className="p-4 rounded-xl bg-green-50"><b className="text-sm text-green-700">Strengths</b><ul className="text-sm text-[#4a5368] mt-1 list-disc pl-4">{a.pros.map((x: string) => <li key={x}>{x}</li>)}</ul></div>
            <div className="p-4 rounded-xl bg-red-50"><b className="text-sm text-red-700">Limitations</b><ul className="text-sm text-[#4a5368] mt-1 list-disc pl-4">{a.cons.map((x: string) => <li key={x}>{x}</li>)}</ul></div>
          </div>
        </div>
        <div className="card p-6">
          <h3 className="font-bold mb-3">Compare algorithms</h3>
          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-3">
            {Object.entries(ALGOS).map(([key, v]: any) => (
              <button key={key} onClick={() => p.setAlgorithm(key)} className={`p-3 rounded-xl text-left border transition ${p.algorithm === key ? "border-[#1e2a44] bg-[#f8f3e8]" : "hover:bg-[#f8f3e8]"}`}>
                <b className="text-sm">{v.name}</b><span className="block text-xs text-[#5b6478]">{v.type}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Evaluation({ data, result, k, algorithm, useBestK, go }: any) {
  if (!result && !data.length) return <Empty text="Run an algorithm first." action={<button className="btn btn-primary" onClick={() => go("Clustering")}>Go to Clustering</button>} />;
  const withSil = data.filter((d: any) => typeof d.silhouette === "number");
  const bestK = withSil.length ? withSil.reduce((b: any, d: any) => (d.silhouette > b.silhouette ? d : b)).k : null;
  return (
    <div className="space-y-5">
      {bestK !== null && (
        <div className="card p-5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3"><Lightbulb className="text-amber-500" /><span className="text-sm">The silhouette score peaks at <b>K = {bestK}</b>. You are currently using K = {k}.</span></div>
          {bestK !== k && algorithm !== "dbscan" && <button className="btn btn-primary text-sm" onClick={() => useBestK(bestK)}>Re-run with K = {bestK}</button>}
        </div>
      )}
      <div className="grid lg:grid-cols-2 gap-5">
        <div className="card p-6">
          <h2 className="font-bold">Elbow method<Tip text="Look for the 'elbow' where adding clusters stops reducing SSE much." /></h2>
          <p className="text-xs text-[#5b6478] mb-4">Inertia (SSE) versus K — lower is tighter</p>
          {data.length ? (
            <ResponsiveContainer width="100%" height={280}><LineChart data={data}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="k" /><YAxis /><Tooltip /><Line type="monotone" dataKey="inertia" stroke="#1e3a5f" strokeWidth={2} dot /></LineChart></ResponsiveContainer>
          ) : <p className="text-[#8b93a5]">No data.</p>}
        </div>
        <div className="card p-6">
          <h2 className="font-bold">Silhouette analysis<Tip text="Higher is better. The best K maximizes this curve." /></h2>
          <p className="text-xs text-[#5b6478] mb-4">Silhouette versus K — higher is better</p>
          {data.length ? (
            <ResponsiveContainer width="100%" height={280}><LineChart data={data}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="k" /><YAxis domain={[0, 1]} /><Tooltip /><Line type="monotone" dataKey="silhouette" stroke="#6b8e4e" strokeWidth={2} dot /></LineChart></ResponsiveContainer>
          ) : <p className="text-[#8b93a5]">No data.</p>}
        </div>
      </div>
      {result && <MetricGrid metrics={result.metrics} />}
    </div>
  );
}

function MetricGrid({ metrics }: any) {
  return (
    <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-4">
      {Object.entries(metrics || {}).map(([key, v]: any) => (
        <div className="card p-5" key={key}>
          <p className="text-sm text-[#8b93a5] capitalize">{key.replaceAll("_", " ")}</p>
          <b className="text-2xl">{prettyNum(v)}</b>
          {key === "silhouette" && <p className={`text-xs mt-1 ${silhouetteLabel(v)[1]}`}>{silhouetteLabel(v)[0]}</p>}
          {GLOSS[key] && <p className="text-xs text-[#5b6478] mt-2">{GLOSS[key]}</p>}
        </div>
      ))}
    </div>
  );
}

function getPoints(pca: any): { x: number; y: number }[] {
  const raw = pca?.points || pca?.coordinates || pca?.components || pca?.coords || pca?.data;
  if (!Array.isArray(raw)) return [];
  return raw.slice(0, 3000).map((p: any) => (Array.isArray(p) ? { x: p[0], y: p[1] } : { x: p.x ?? p.PC1 ?? p.pc1, y: p.y ?? p.PC2 ?? p.pc2 }));
}

function Results({ result, pca, history, exportResult, go }: any) {
  if (!result) return <Empty text="Run clustering first." action={<button className="btn btn-primary" onClick={() => go("Clustering")}>Go to Clustering</button>} />;
  const counts = Object.entries(result.cluster_counts || {}).map(([cluster, count]) => ({ cluster, count: count as number }));
  const total = counts.reduce((a, c) => a + c.count, 0) || 1;
  const colorOf = (c: any) => (String(c) === "-1" || /noise/i.test(String(c)) ? NOISE_COLOR : COLORS[Math.abs(parseInt(String(c).replace(/\D/g, "") || "0")) % COLORS.length]);
  const pts = getPoints(pca);
  const groups: Record<string, any[]> = {};
  pts.forEach((p, i) => { const l = String(result.labels?.[i] ?? 0); (groups[l] = groups[l] || []).push(p); });
  const sil = result.metrics?.silhouette;
  const largest = [...counts].sort((a, b) => b.count - a.count)[0];
  const smallest = [...counts].sort((a, b) => a.count - b.count)[0];
  const noise = counts.find((c) => String(c.cluster) === "-1" || /noise/i.test(String(c.cluster)));

  return (
    <div className="space-y-5">
      <div className="card p-6 bg-[#1e2a44] text-white">
        <h3 className="font-bold flex items-center gap-2"><Sparkles size={16} />Auto-generated insights</h3>
        <ul className="text-sm mt-3 space-y-1.5 text-gray-200 list-disc pl-5">
          <li>{ALGOS[result.algorithm]?.name || result.algorithm} found <b>{counts.length - (noise ? 1 : 0)}</b> clusters across <b>{total}</b> records.</li>
          {largest && <li>The largest group (cluster {largest.cluster}) holds {Math.round((largest.count / total) * 100)}% of the data; the smallest (cluster {smallest.cluster}) holds {Math.round((smallest.count / total) * 100)}%.</li>}
          {noise && <li>{noise.count} records ({Math.round((noise.count / total) * 100)}%) were flagged as noise/outliers.</li>}
          <li>Cluster quality: <b>{silhouetteLabel(sil)[0]}</b>{typeof sil === "number" ? ` (silhouette ${sil.toFixed(3)})` : ""}.{typeof sil === "number" && sil < 0.25 ? " Try a different K, algorithm or scaler." : ""}</li>
        </ul>
      </div>

      <div className="flex flex-wrap justify-between gap-3">
        <button className="btn" onClick={() => go("Clustering")}><RotateCcw size={16} className="inline mr-2" />Tune &amp; re-run</button>
        <button className="btn btn-primary" onClick={exportResult}><Download size={16} className="inline mr-2" />Export CSV</button>
      </div>

      <MetricGrid metrics={result.metrics} />

      <div className="grid lg:grid-cols-2 gap-5">
        <div className="card p-6">
          <h3 className="font-bold mb-5">Cluster distribution</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={counts}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="cluster" /><YAxis /><Tooltip />
              <Bar dataKey="count">{counts.map((c) => <Cell key={c.cluster} fill={colorOf(c.cluster)} />)}</Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card p-6">
          <h3 className="font-bold mb-3">PCA explained variance<Tip text="Share of the data's variance captured by each principal component." /></h3>
          {pca?.explained_variance ? (
            <div className="space-y-3">
              {pca.explained_variance.map((x: number, i: number) => (
                <div key={i}>
                  <div className="flex justify-between text-sm"><span>PC{i + 1}</span><b>{(x * 100).toFixed(1)}%</b></div>
                  <div className="h-2 bg-[#efe8d8] rounded mt-1"><div className="h-2 bg-[#1e2a44] rounded" style={{ width: `${x * 100}%` }} /></div>
                </div>
              ))}
              <p className="text-xs text-[#5b6478]">The first two components capture {(((pca.explained_variance[0] || 0) + (pca.explained_variance[1] || 0)) * 100).toFixed(1)}% of the variance.</p>
            </div>
          ) : <p className="text-[#8b93a5]">No PCA data.</p>}
        </div>
      </div>

      <div className="card p-6">
        <h3 className="font-bold mb-1">Cluster map (PCA projection)</h3>
        <p className="text-xs text-[#5b6478] mb-4">Each dot is a record, projected to 2D and coloured by cluster.</p>
        {pts.length ? (
          <ResponsiveContainer width="100%" height={380}>
            <ScatterChart><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" dataKey="x" name="PC1" /><YAxis type="number" dataKey="y" name="PC2" /><Tooltip cursor={{ strokeDasharray: "3 3" }} /><Legend />
              {Object.entries(groups).map(([l, arr]) => <Scatter key={l} name={l === "-1" ? "Noise" : `Cluster ${l}`} data={arr} fill={colorOf(l)} />)}
            </ScatterChart>
          </ResponsiveContainer>
        ) : <p className="text-sm text-[#8b93a5]">Your /api/pca endpoint doesn't return 2D coordinates yet. Add a <code>points</code> list of [PC1, PC2] pairs (see notes) to enable this chart.</p>}
      </div>

      {history.length > 0 && (
        <div className="card p-6">
          <h3 className="font-bold mb-3">Run history &amp; comparison</h3>
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead><tr className="text-left text-[#8b93a5] text-xs"><th className="py-2">#</th><th>Algorithm</th><th>Parameters</th><th>Preprocessing</th><th>Clusters</th><th>Silhouette</th></tr></thead>
              <tbody>
                {history.map((h: Run) => {
                  const best = Math.max(...history.map((x: Run) => x.silhouette ?? -2));
                  return (
                    <tr key={h.id} className="border-t">
                      <td className="py-2">{h.id}</td><td>{ALGOS[h.algorithm]?.name}</td>
                      <td>{h.eps !== undefined ? `eps=${h.eps}, min=${h.minSamples}` : `K=${h.k}`}</td>
                      <td className="text-[#5b6478]">{h.strategy} / {h.scaler}</td><td>{h.clusters}</td>
                      <td className={h.silhouette === best ? "font-bold text-green-600" : ""}>{h.silhouette === null ? "–" : h.silhouette.toFixed(3)}{h.silhouette === best && " ★"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

/* ----------------------------- Learn: concepts + playground ----------------------------- */

type Pt = { x: number; y: number };
const dist2 = (a: Pt, b: Pt) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
const assignPts = (pts: Pt[], c: Pt[]) => pts.map((p) => c.reduce((bi, cc, i) => (dist2(p, cc) < dist2(p, c[bi]) ? i : bi), 0));
const updateCent = (pts: Pt[], a: number[], c: Pt[]): Pt[] =>
  c.map((old, i) => {
    const m = pts.filter((_, j) => a[j] === i);
    return m.length ? { x: m.reduce((s, p) => s + p.x, 0) / m.length, y: m.reduce((s, p) => s + p.y, 0) / m.length } : old;
  });

function Playground() {
  const gen = (): Pt[] => {
    const cs = [[110, 100], [440, 90], [270, 270]];
    return Array.from({ length: 90 }, (_, i) => ({ x: cs[i % 3][0] + (Math.random() - 0.5) * 140, y: cs[i % 3][1] + (Math.random() - 0.5) * 120 }));
  };
  const [pts, setPts] = useState<Pt[]>(gen);
  const [k, setK] = useState(3);
  const [cent, setCent] = useState<Pt[]>([]);
  const [asg, setAsg] = useState<number[]>([]);
  const [phase, setPhase] = useState<"assign" | "update">("assign");
  const [iter, setIter] = useState(0);
  const [msg, setMsg] = useState("Step 1: pick K random points as initial centroids.");
  const svg = useRef<SVGSVGElement>(null);

  const init = () => {
    setCent([...pts].sort(() => Math.random() - 0.5).slice(0, k));
    setAsg([]); setPhase("assign"); setIter(0);
    setMsg("Centroids placed randomly. Click “Next step” to assign each point to its nearest centroid.");
  };
  useEffect(init, [k, pts]);

  const step = () => {
    if (!cent.length) return;
    if (phase === "assign") {
      setAsg(assignPts(pts, cent)); setPhase("update");
      setMsg("Assignment: every point joined its nearest centroid (Euclidean distance).");
    } else {
      setCent(updateCent(pts, asg, cent)); setPhase("assign"); setIter((i) => i + 1);
      setMsg("Update: each centroid moved to the mean of its points. Repeat until nothing changes.");
    }
  };
  const converge = () => {
    let c = cent, a: number[] = [], n = iter;
    for (let i = 0; i < 40; i++) {
      a = assignPts(pts, c);
      const nc = updateCent(pts, a, c);
      n++;
      const moved = nc.some((p, j) => dist2(p, c[j]) > 0.01);
      c = nc;
      if (!moved) break;
    }
    setCent(c); setAsg(assignPts(pts, c)); setIter(n); setPhase("assign");
    setMsg("Converged: centroids stopped moving, so the algorithm is finished.");
  };
  const addPoint = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = svg.current!.getBoundingClientRect();
    setPts((p) => [...p, { x: ((e.clientX - r.left) / r.width) * 600, y: ((e.clientY - r.top) / r.height) * 360 }]);
  };
  const sse = asg.length ? pts.reduce((s, p, i) => s + dist2(p, cent[asg[i]]), 0) : null;

  return (
    <div className="card p-6">
      <h2 className="font-bold text-lg">K-Means playground</h2>
      <p className="text-sm text-[#5b6478] mb-4">Click the canvas to add points, then step through the algorithm and watch the centroids move.</p>
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <label className="text-sm flex items-center gap-2">K = <b>{k}</b><input type="range" min={2} max={6} value={k} onChange={(e) => setK(+e.target.value)} /></label>
        <button className="btn text-xs" onClick={step}><Play size={14} className="inline mr-1" />Next step ({phase === "assign" ? "assign" : "update"})</button>
        <button className="btn btn-primary text-xs" onClick={converge}>Run to convergence</button>
        <button className="btn text-xs" onClick={init}><RotateCcw size={14} className="inline mr-1" />Reset centroids</button>
        <button className="btn text-xs" onClick={() => setPts(gen())}><Shuffle size={14} className="inline mr-1" />New data</button>
      </div>
      <svg ref={svg} viewBox="0 0 600 360" className="w-full bg-[#f8f3e8] rounded-xl border cursor-crosshair" onClick={addPoint}>
        {pts.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r={4.5} fill={asg[i] !== undefined ? COLORS[asg[i] % COLORS.length] : "#9ca3af"} opacity={0.8} />)}
        {cent.map((c, i) => <g key={i}><circle cx={c.x} cy={c.y} r={11} fill={COLORS[i % COLORS.length]} stroke="#111827" strokeWidth={2.5} /><text x={c.x} y={c.y + 4} textAnchor="middle" fontSize="11" fill="#fff" fontWeight="bold">{i + 1}</text></g>)}
      </svg>
      <div className="flex flex-wrap justify-between gap-2 mt-3 text-sm">
        <span className="text-[#4a5368]">{msg}</span>
        <span className="text-[#5b6478]">Iterations: <b>{iter}</b>{sse !== null && <> · SSE: <b>{sse.toFixed(0)}</b></>}</span>
      </div>
    </div>
  );
}

function Learn() {
  const [open, setOpen] = useState<number | null>(0);
  return (
    <div className="space-y-5">
      <Playground />
      <div className="card p-6">
        <h2 className="font-bold text-lg mb-4">DWM concept guide</h2>
        <div className="space-y-2">
          {CONCEPTS.map(([t, d], i) => (
            <div key={t} className="border rounded-xl">
              <button className="w-full flex justify-between items-center p-3 text-left" onClick={() => setOpen(open === i ? null : i)}>
                <b className="text-sm">{t}</b><span className="text-[#8b93a5]">{open === i ? "−" : "+"}</span>
              </button>
              {open === i && <p className="px-3 pb-3 text-sm text-[#4a5368]">{d}</p>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
