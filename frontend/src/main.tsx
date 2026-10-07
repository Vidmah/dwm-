import React,{useState} from "react";
import {createRoot} from "react-dom/client";
import {Upload,Database,SlidersHorizontal,BrainCircuit,BarChart3,Download,CheckCircle2,Play,FileSpreadsheet} from "lucide-react";
import {LineChart,Line,XAxis,YAxis,CartesianGrid,Tooltip,ResponsiveContainer,BarChart,Bar} from "recharts";
import "./index.css";

const API = `${import.meta.env.VITE_API_URL}/api`;
type Info={dataset_id:string,filename:string,rows:number,columns:number,columns_info:any[],preview:any[]};
type Result={algorithm:string,metrics:any,cluster_counts:any,labels:number[],centers?:any,medoids?:any};

function App(){
 const [tab,setTab]=useState("Dashboard"); const [info,setInfo]=useState<Info|null>(null);
 const [features,setFeatures]=useState<string[]>([]); const [strategy,setStrategy]=useState("median"); const [scaler,setScaler]=useState("standard");
 const [algorithm,setAlgorithm]=useState("kmeans"); const [k,setK]=useState(3); const [eps,setEps]=useState(.5); const [minSamples,setMinSamples]=useState(5);
 const [result,setResult]=useState<Result|null>(null); const [evalData,setEvalData]=useState<any[]>([]); const [pca,setPca]=useState<any|null>(null); const [busy,setBusy]=useState(false); const [error,setError]=useState("");

 async function upload(file:File){
  setBusy(true);setError(""); const fd=new FormData();fd.append("file",file);
  try{const r=await fetch(API+"/upload",{method:"POST",body:fd});if(!r.ok)throw Error(await r.text());const d=await r.json();setInfo(d);setFeatures(d.columns_info.filter((x:any)=>["int64","float64","int32","float32"].includes(x.dtype)).map((x:any)=>x.name));setTab("Preprocess")}
  catch(e:any){setError(e.message)}finally{setBusy(false)}
 }
 const payload=()=>({dataset_id:info?.dataset_id,features,missing_strategy:strategy,scaler,k,eps,min_samples:minSamples,algorithm,linkage:"ward"});
 async function run(){
  if(!info||!features.length)return;setBusy(true);setError("");
  try{const r=await fetch(API+"/cluster",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload())});if(!r.ok)throw Error(await r.text());setResult(await r.json());
   const e=await fetch(API+"/evaluate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload())});setEvalData((await e.json()).scores);
   const p=await fetch(API+"/pca",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload())});setPca(await p.json());setTab("Results")}
  catch(e:any){setError(e.message)}finally{setBusy(false)}
 }
 async function exportResult(){if(!info)return;const r=await fetch(API+"/export",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload())});const b=await r.blob();const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="clustering_results.csv";a.click()}
 const nav=[["Dashboard",Database],["Upload",Upload],["Preprocess",SlidersHorizontal],["Clustering",BrainCircuit],["Evaluation",BarChart3],["Results",CheckCircle2]];
 return <div className="min-h-screen flex bg-[#f7f8fa]">
  <aside className="w-64 bg-white border-r border-gray-200 p-5 hidden md:block">
   <div className="flex items-center gap-2 mb-8"><div className="bg-gray-900 text-white p-2 rounded-xl"><BrainCircuit size={20}/></div><div><b>DataMine</b><p className="text-xs text-gray-400">Studio</p></div></div>
   <div className="space-y-1">{nav.map(([n,I]:any)=><button key={n} onClick={()=>setTab(n)} className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm ${tab===n?"bg-gray-900 text-white":"text-gray-600 hover:bg-gray-100"}`}><I size={17}/>{n}</button>)}</div>
   <div className="mt-8 p-4 rounded-xl bg-gray-50 text-xs text-gray-500">Educational Data Mining Dashboard<br/><span className="text-gray-800 font-semibold">5 clustering algorithms</span></div>
  </aside>
  <main className="flex-1 p-5 md:p-8 max-w-[1500px] mx-auto w-full">
   <header className="flex justify-between items-center mb-7"><div><h1 className="text-2xl font-bold text-gray-900">{tab}</h1><p className="text-sm text-gray-500">Analyze, cluster and evaluate your dataset.</p></div>{info&&<div className="hidden sm:flex items-center gap-2 text-sm bg-white border px-3 py-2 rounded-xl"><FileSpreadsheet size={16}/>{info.filename}</div>}</header>
   {error&&<div className="mb-5 p-3 rounded-xl bg-red-50 text-red-700 text-sm">{error}</div>}
   {tab==="Dashboard"&&<Dashboard info={info} result={result} evalData={evalData} onUpload={upload}/>}
   {tab==="Upload"&&<UploadPage onUpload={upload} busy={busy} info={info}/>}
   {tab==="Preprocess"&&<Preprocess info={info} features={features} setFeatures={setFeatures} strategy={strategy} setStrategy={setStrategy} scaler={scaler} setScaler={setScaler}/>}
   {tab==="Clustering"&&<ClusterPage info={info} algorithm={algorithm} setAlgorithm={setAlgorithm} k={k} setK={setK} eps={eps} setEps={setEps} minSamples={minSamples} setMinSamples={setMinSamples} run={run} busy={busy}/>}
   {tab==="Evaluation"&&<Evaluation data={evalData} result={result}/>}
   {tab==="Results"&&<Results result={result} pca={pca} exportResult={exportResult}/>}
  </main>
 </div>
}

function Dashboard({info,result,evalData,onUpload}:any){return <div className="space-y-6">
 <div className="card p-7"><div className="max-w-2xl"><span className="text-xs font-bold uppercase tracking-widest text-gray-400">Data Mining Laboratory</span><h2 className="text-4xl font-bold mt-2">Turn raw data into<br/><span className="text-gray-500">discoverable patterns.</span></h2><p className="text-gray-500 mt-4">Upload a dataset, preprocess it, apply clustering algorithms and compare the results.</p><button className="btn btn-primary mt-6" onClick={()=>document.getElementById("dashfile")?.click()}><Upload size={16} className="inline mr-2"/>Upload Dataset</button><input id="dashfile" type="file" accept=".csv,.xlsx,.xls" hidden onChange={e=>e.target.files&&onUpload(e.target.files[0])}/></div></div>
 <div className="grid md:grid-cols-4 gap-4">{[["Rows",info?.rows||0],["Columns",info?.columns||0],["Numeric features",info?.columns_info?.filter((x:any)=>x.dtype.includes("int")||x.dtype.includes("float")).length||0],["Missing values",info?.columns_info?.reduce((a:number,x:any)=>a+x.missing,0)||0]].map(x=><div className="card p-5" key={x[0]}><p className="text-sm text-gray-400">{x[0]}</p><b className="text-3xl">{x[1]}</b></div>)}</div>
 <div className="grid lg:grid-cols-2 gap-5"><div className="card p-6"><h3 className="font-bold mb-4">Pipeline</h3>{["Upload","Preprocess","Cluster","Evaluate","Export"].map((x,i)=><div className="flex items-center gap-3 py-2" key={x}><span className="w-7 h-7 rounded-full bg-gray-100 flex items-center justify-center text-xs">{i+1}</span>{x}</div>)}</div><div className="card p-6"><h3 className="font-bold mb-4">Latest model</h3>{result?<><p className="text-2xl font-bold">{result.algorithm}</p><p className="text-gray-500 mt-2">Silhouette: <b>{result.metrics?.silhouette??"N/A"}</b></p></>:<p className="text-gray-400">Run a clustering algorithm to see results.</p>}</div></div>
 </div>}

function UploadPage({onUpload,busy,info}:any){return <div className="card p-8"><div className="border-2 border-dashed border-gray-300 rounded-2xl p-16 text-center"><Upload className="mx-auto mb-4 text-gray-400" size={38}/><h2 className="text-xl font-bold">Drop CSV or Excel file here</h2><p className="text-gray-500 text-sm mt-2">Supported: .csv, .xlsx, .xls</p><label className="btn btn-primary inline-block mt-5 cursor-pointer">{busy?"Uploading...":"Choose file"}<input type="file" hidden accept=".csv,.xlsx,.xls" onChange={e=>e.target.files&&onUpload(e.target.files[0])}/></label></div>{info&&<div className="mt-6"><b>{info.filename}</b><p className="text-sm text-gray-500">{info.rows} rows × {info.columns} columns</p></div>}</div>}

function Preprocess({info,features,setFeatures,strategy,setStrategy,scaler,setScaler}:any){if(!info)return <Empty text="Upload a dataset first."/>;return <div className="space-y-5"><div className="card p-6"><h2 className="font-bold text-lg">Feature selection</h2><p className="text-sm text-gray-500 mb-5">Choose numerical columns used by clustering and PCA.</p><div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-2">{info.columns_info.map((c:any)=><label key={c.name} className="border rounded-xl p-3 flex gap-3 items-center"><input type="checkbox" checked={features.includes(c.name)} disabled={!["int64","float64","int32","float32"].includes(c.dtype)} onChange={()=>setFeatures((f:string[])=>f.includes(c.name)?f.filter(x=>x!==c.name):[...f,c.name])}/><span className="text-sm">{c.name}<small className="block text-gray-400">{c.dtype} · {c.missing} missing</small></span></label>)}</div></div><div className="grid md:grid-cols-2 gap-5"><div className="card p-6"><h3 className="font-bold">Missing values</h3><select className="mt-4 w-full" value={strategy} onChange={e=>setStrategy(e.target.value)}><option value="median">Median</option><option value="mean">Mean</option><option value="mode">Mode</option><option value="drop">Drop rows</option></select></div><div className="card p-6"><h3 className="font-bold">Normalization</h3><select className="mt-4 w-full" value={scaler} onChange={e=>setScaler(e.target.value)}><option value="standard">StandardScaler</option><option value="minmax">Min-Max</option><option value="robust">RobustScaler</option></select></div></div></div>}

function ClusterPage(p:any){if(!p.info)return <Empty text="Upload a dataset first."/>;return <div className="grid lg:grid-cols-3 gap-5"><div className="card p-6 lg:col-span-1"><h2 className="font-bold text-lg">Algorithm</h2><select className="w-full mt-4" value={p.algorithm} onChange={e=>p.setAlgorithm(e.target.value)}>{["kmeans","kmedoids","dbscan","agnes","diana"].map((x:string)=><option key={x}>{x}</option>)}</select>{p.algorithm==="dbscan"?<><label className="block text-sm mt-5">Epsilon<input className="w-full mt-1" type="number" step=".1" value={p.eps} onChange={e=>p.setEps(+e.target.value)}/></label><label className="block text-sm mt-4">Min samples<input className="w-full mt-1" type="number" value={p.minSamples} onChange={e=>p.setMinSamples(+e.target.value)}/></label></>:<label className="block text-sm mt-5">Number of clusters (K)<input className="w-full mt-1" type="number" min="2" value={p.k} onChange={e=>p.setK(+e.target.value)}/></label>}<button className="btn btn-primary w-full mt-7" onClick={p.run} disabled={p.busy}><Play size={16} className="inline mr-2"/>{p.busy?"Running...":"Run algorithm"}</button></div><div className="card p-6 lg:col-span-2"><h2 className="font-bold text-lg">Algorithm guide</h2><div className="grid md:grid-cols-2 gap-4 mt-5">{[["K-Means","Centroid-based partitioning"],["K-Medoids","Uses representative data points"],["DBSCAN","Density-based; detects noise"],["AGNES","Agglomerative hierarchical clustering"],["DIANA","Divisive hierarchical clustering"]].map(x=><div className="p-4 rounded-xl bg-gray-50" key={x[0]}><b>{x[0]}</b><p className="text-sm text-gray-500 mt-1">{x[1]}</p></div>)}</div></div></div>}

function Evaluation({data,result}:any){return <div className="space-y-5"><div className="card p-6"><h2 className="font-bold mb-5">Elbow & Silhouette analysis</h2>{data.length?<ResponsiveContainer width="100%" height={330}><LineChart data={data}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="k"/><YAxis/><Tooltip/><Line type="monotone" dataKey="inertia" strokeWidth={2}/><Line type="monotone" dataKey="silhouette" strokeWidth={2}/></LineChart></ResponsiveContainer>:<Empty text="Run an algorithm first."/ >}</div>{result&&<div className="grid md:grid-cols-3 gap-4">{Object.entries(result.metrics||{}).map(([k,v]:any)=><div className="card p-5" key={k}><p className="text-sm text-gray-400">{k.replaceAll("_"," ")}</p><b className="text-2xl">{v}</b></div>)}</div>}</div>}

function Results({result,pca,exportResult}:any){if(!result)return <Empty text="Run clustering first."/>;const counts=Object.entries(result.cluster_counts||{}).map(([cluster,count])=>({cluster,count}));return <div className="space-y-5"><div className="flex justify-end"><button className="btn btn-primary" onClick={exportResult}><Download size={16} className="inline mr-2"/>Export CSV</button></div><div className="grid md:grid-cols-3 gap-4">{Object.entries(result.metrics||{}).map(([k,v]:any)=><div className="card p-5" key={k}><p className="text-sm text-gray-400 capitalize">{k.replaceAll("_"," ")}</p><b className="text-2xl">{v}</b></div>)}</div><div className="grid lg:grid-cols-2 gap-5"><div className="card p-6"><h3 className="font-bold mb-5">Cluster distribution</h3><ResponsiveContainer width="100%" height={280}><BarChart data={counts}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="cluster"/><YAxis/><Tooltip/><Bar dataKey="count"/></BarChart></ResponsiveContainer></div><div className="card p-6"><h3 className="font-bold mb-3">PCA explained variance</h3>{pca?<div className="space-y-3">{pca.explained_variance.map((x:number,i:number)=><div key={i}><div className="flex justify-between text-sm"><span>PC{i+1}</span><b>{(x*100).toFixed(1)}%</b></div><div className="h-2 bg-gray-100 rounded mt-1"><div className="h-2 bg-gray-900 rounded" style={{width:`${x*100}%`}}/></div></div>)}</div>:<p className="text-gray-400">No PCA data.</p>}</div></div></div>}

function Empty({text}:any){return <div className="card p-10 text-center text-gray-500">{text}</div>}
createRoot(document.getElementById("root")!).render(<App/>);
