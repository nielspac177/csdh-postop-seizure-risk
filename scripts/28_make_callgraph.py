"""Task 28 — Lightweight callgraph generator.

Parses every script in scripts/ via `ast` and produces:
  • a Mermaid call-graph diagram (markdown + svg-friendly)
  • a per-script function inventory in CALLGRAPH.md

This avoids heavy tools like pyan3 (which require call-site type inference);
ast-level parsing is sufficient for "which module imports / calls what" and
"what does each function compute".
"""
import sys, os, ast, re
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_DIR    = SCRIPTS_DIR.parent / "github_repo"
OUT_PATH    = REPO_DIR / "CALLGRAPH.md"

def parse_script(path):
    """Returns dict with: imports (list), functions (list of (name, args, docstring1))."""
    try:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
    except Exception as e:
        return {"imports": [], "functions": [], "error": str(e)}
    imports = []
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            doc1 = doc.split("\n")[0][:160] if doc else ""
            args = [a.arg for a in node.args.args]
            functions.append((node.name, args, doc1))
    imports = sorted(set(imports))
    return {"imports": imports, "functions": functions}


def make_mermaid(scripts):
    """Build a Mermaid graph: nodes are scripts, edges are imports between them."""
    lines = ["```mermaid", "graph LR",
              "    classDef shared fill:#fff7e0,stroke:#aa7700,stroke-width:2px;",
              "    classDef model fill:#e6f3ff,stroke:#1f4e79;",
              "    classDef cea   fill:#fef2dc,stroke:#d4621a;",
              "    classDef doc   fill:#e8f7e8,stroke:#2a8a3f;",
              ""]
    # nodes
    node_ids = {}
    for name in scripts:
        nid = "S" + re.sub(r"\W", "_", name.replace(".py", ""))
        node_ids[name] = nid
        label = name
        lines.append(f'    {nid}["{label}"]')
    # group classes
    shared = [n for n in scripts if n.startswith("_shared")]
    model  = [n for n in scripts if any(s in n for s in ("18", "19", "21", "22", "23",
                                                          "24", "25", "06", "02"))]
    cea    = [n for n in scripts if any(s in n for s in ("10_11", "14", "16"))]
    docs   = [n for n in scripts if any(s in n for s in ("17", "20", "26", "27"))]
    if shared:
        lines.append("    class " + ",".join(node_ids[n] for n in shared) + " shared;")
    if model:
        lines.append("    class " + ",".join(node_ids[n] for n in model) + " model;")
    if cea:
        lines.append("    class " + ",".join(node_ids[n] for n in cea) + " cea;")
    if docs:
        lines.append("    class " + ",".join(node_ids[n] for n in docs) + " doc;")
    # edges
    for name, info in scripts.items():
        for imp in info["imports"]:
            if imp == "_shared":
                # mark dependency on the shared module
                if "_shared.py" in node_ids:
                    lines.append(f'    {node_ids[name]} --> {node_ids["_shared.py"]}')
            # script-to-script imports (only when script name is referenced)
            for other in scripts:
                base = other.replace(".py", "")
                if base == imp:
                    lines.append(f'    {node_ids[name]} --> {node_ids[other]}')
    lines.append("```")
    return "\n".join(lines)


def main():
    scripts = {}
    for p in sorted(SCRIPTS_DIR.glob("*.py")):
        scripts[p.name] = parse_script(p)

    sections = [
        "# Callgraph and module inventory",
        "",
        "## Visual dependency graph",
        "",
        "Nodes are scripts; arrows are import dependencies. Shared utilities "
        "(`_shared.py`) sit at the root and feed every analysis module. "
        "Modelling scripts read cached out-of-fold predictions emitted by "
        "calibration (02), and the manuscript builders (20, 27) consume the "
        "downstream CSV outputs.",
        "",
        make_mermaid(scripts),
        "",
        "## Per-script function inventory",
        "",
        "Each script is single-purpose, executable from the command line, "
        "deterministic (SEED=42, n_jobs=1), and produces CSV results and "
        "PNG/PDF figures in fixed output paths.",
        "",
    ]
    for name in sorted(scripts):
        info = scripts[name]
        sections.append(f"### `{name}`")
        if not info["functions"] and not info["imports"]:
            sections.append("_(empty module)_"); sections.append(""); continue
        if info.get("error"):
            sections.append(f"**Parse error:** {info['error']}")
            sections.append(""); continue
        # top imports
        notable = [i for i in info["imports"]
                    if i not in {"os", "sys", "warnings", "json", "pathlib", "_shared"}]
        if notable:
            sections.append(f"**Imports:** {', '.join(notable)}")
        # functions
        if info["functions"]:
            sections.append("")
            sections.append("| Function | Args | Purpose |")
            sections.append("|---|---|---|")
            for fname, args, doc in info["functions"]:
                arg_str = ", ".join(args)
                if len(arg_str) > 60:
                    arg_str = arg_str[:57] + "..."
                doc_clean = doc.replace("|", "\\|") if doc else "—"
                sections.append(f"| `{fname}` | `{arg_str}` | {doc_clean} |")
        sections.append("")

    OUT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"[OK] {OUT_PATH}")
    print(f"     scripts parsed: {len(scripts)}")
    print(f"     total functions documented: "
          f"{sum(len(s['functions']) for s in scripts.values())}")


if __name__ == "__main__":
    main()
