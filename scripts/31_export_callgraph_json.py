"""Task 31 — Export the module dependency graph as JSON for the web callgraph.

The static HTML page at site/callgraph.html consumes site/callgraph.json
to render a node-link visualisation via vis-network. Nodes are scripts;
edges are import relationships; each node carries its function inventory
in a tooltip / side panel.
"""
import sys, os, ast, json, re
sys.path.insert(0, os.path.dirname(__file__))

SCRIPTS_DIR = os.path.dirname(__file__)
REPO_SITE = os.path.normpath(
    os.path.join(SCRIPTS_DIR, "..", "github_repo", "site")
)
os.makedirs(REPO_SITE, exist_ok=True)
OUT_PATH = os.path.join(REPO_SITE, "callgraph.json")

CATEGORY = {
    "_shared.py": "shared",
    "02_calibration.py": "model",
    "03_dca.py": "model",
    "04_loho.py": "model",
    "05_temporal_leakage.py": "model",
    "06_overfitting.py": "model",
    "07_missing_data.py": "model",
    "08_eicu_cohort.py": "model",
    "09_competing_risks.py": "model",
    "10_11_cea_pairwise.py": "cea",
    "12_nis_seizure_reclassify.py": "model",
    "13_nis_grouped_lasso.py": "model",
    "14_decision_tree.py": "cea",
    "15_radiology_nlp.py": "model",
    "16_voi_evpi.py": "cea",
    "17_build_slides.py": "doc",
    "18_bidmc_optimize.py": "model",
    "19_transfer_learning.py": "model",
    "20_build_manuscript.py": "doc",
    "21_imbalance_sweep.py": "model",
    "22_diverse_stacking.py": "model",
    "23_tabpfn_eval.py": "model",
    "24_firth_bayes_lr.py": "model",
    "25_conformal_prediction.py": "model",
    "26_main_figures.py": "doc",
    "27_build_jnnp_manuscript.py": "doc",
    "28_make_callgraph.py": "doc",
    "29_main_figures_jnnp.py": "doc",
    "30_export_calculator_assets.py": "doc",
    "31_export_callgraph_json.py": "doc",
}

def parse_script(path):
    try:
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src)
    except Exception as e:
        return {"imports": [], "functions": [], "error": str(e)}
    imports = set(); functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            doc1 = doc.split("\n")[0][:140] if doc else ""
            args = [a.arg for a in node.args.args]
            functions.append({"name": node.name, "args": args, "doc": doc1})
    return {"imports": sorted(imports), "functions": functions}


def main():
    nodes, edges = [], []
    files = sorted(os.listdir(SCRIPTS_DIR))
    py_files = [f for f in files if f.endswith(".py")]
    by_name = {}
    for f in py_files:
        info = parse_script(os.path.join(SCRIPTS_DIR, f))
        info["category"] = CATEGORY.get(f, "model")
        by_name[f] = info
    for f, info in by_name.items():
        n_funcs = len(info["functions"])
        nodes.append({
            "id": f,
            "label": f.replace(".py", ""),
            "title": (f"{f}\n{n_funcs} functions\n" +
                       "category: " + info["category"]),
            "category": info["category"],
            "n_functions": n_funcs,
            "functions": info["functions"],
        })
    # edges: detect imports that match other module bases
    bases = {f.replace(".py", ""): f for f in py_files}
    for f, info in by_name.items():
        for imp in info["imports"]:
            if imp == "_shared":
                edges.append({"from": f, "to": "_shared.py"})
            elif imp in bases and bases[imp] != f:
                edges.append({"from": f, "to": bases[imp]})
    # dedupe edges
    seen = set(); unique_edges = []
    for e in edges:
        k = (e["from"], e["to"])
        if k not in seen:
            seen.add(k); unique_edges.append(e)
    payload = {
        "nodes": nodes,
        "edges": unique_edges,
        "categories": {
            "shared":   {"color": "#B58A2E", "label": "Shared utilities"},
            "model":    {"color": "#1F3D5C", "label": "Modelling"},
            "cea":      {"color": "#B5532C", "label": "Cost-effectiveness / VOI"},
            "doc":      {"color": "#2E6B45", "label": "Manuscript / figures / dashboards"},
        },
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] {OUT_PATH}")
    print(f"     nodes: {len(payload['nodes'])}, edges: {len(payload['edges'])}")


if __name__ == "__main__":
    main()
