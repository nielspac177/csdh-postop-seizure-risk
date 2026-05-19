"""Task 33 — Build the PDF/LaTeX Code Companion.

Writes a single LaTeX file at github_repo/docs/Code_Companion.tex that
mirrors the Word companion but with proper syntax highlighting via the
listings package, a hyperlinked table of contents, sidebar bookmarks
in the PDF reader, and vector typography. Compiles to PDF with xelatex.

The content is written by hand below; the script embeds the actual code
listings by reading the scripts and the dashboard HTML directly so that
the document is always in sync with the codebase.
"""
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path

from _shared import OUT

REPO = OUT / "github_repo"
DOCS = REPO / "docs"
SCRIPTS = REPO / "scripts"
SITE = REPO / "site"

TEX_PATH = DOCS / "Code_Companion.tex"
PDF_PATH = DOCS / "Code_Companion.pdf"
DOCS.mkdir(parents=True, exist_ok=True)


def read_text(path):
    p = Path(path)
    if not p.exists(): return f"% file not found: {path}"
    return p.read_text(encoding="utf-8")


def code_snippet(src, max_lines=None, start_token=None, end_token=None):
    """Optionally trim a source to a section. max_lines caps the length."""
    if start_token is not None:
        i = src.find(start_token)
        if i >= 0:
            src = src[i:]
    if end_token is not None:
        j = src.find(end_token)
        if j >= 0:
            src = src[:j].rstrip()
    if max_lines:
        lines = src.splitlines()
        if len(lines) > max_lines:
            src = "\n".join(lines[:max_lines]) + (
                f"\n... ({len(lines) - max_lines} more lines elided "
                "— see full script in the repository) ...")
    return src


# Read all the source files we will embed
SRC_SHARED   = read_text(SCRIPTS / "_shared.py")
SRC_CAL      = read_text(SCRIPTS / "02_calibration.py")
SRC_LOHO     = read_text(SCRIPTS / "04_loho.py")
SRC_TREE     = read_text(SCRIPTS / "14_decision_tree.py")
SRC_FIRTH    = read_text(SCRIPTS / "24_firth_bayes_lr.py")
SRC_CONF     = read_text(SCRIPTS / "25_conformal_prediction.py")
SRC_VOI      = read_text(SCRIPTS / "16_voi_evpi.py")
SRC_FIG      = read_text(SCRIPTS / "29_main_figures_jnnp.py")
SRC_MANUS    = read_text(SCRIPTS / "27_build_jnnp_manuscript.py")
SRC_EXPORT   = read_text(SCRIPTS / "30_export_calculator_assets.py")
SRC_CALC     = read_text(SITE / "calculator.html")


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

% --- Encoding and fonts (XeLaTeX) ---
\usepackage{fontspec}
\setmainfont{Times}
\setsansfont{Helvetica}
\setmonofont{Menlo}[Scale=0.85]

% --- Layout ---
\usepackage[a4paper,margin=2.4cm]{geometry}
\usepackage{microtype}
\usepackage{parskip}
\setlength{\parindent}{0pt}
\linespread{1.18}

% --- Colours (CMYK-safe palette matching the figures) ---
\usepackage{xcolor}
\definecolor{navy}{HTML}{1F3D5C}
\definecolor{rust}{HTML}{B5532C}
\definecolor{forest}{HTML}{2E6B45}
\definecolor{ochre}{HTML}{B58A2E}
\definecolor{soft}{HTML}{F4EFE6}
\definecolor{grey}{HTML}{6E6E6E}
\definecolor{codebg}{HTML}{F6F6F4}
\definecolor{codeframe}{HTML}{D8D5CB}
\definecolor{kwd}{HTML}{1F4E79}
\definecolor{str}{HTML}{2E6B45}
\definecolor{com}{HTML}{6E6E6E}
\definecolor{num}{HTML}{B5532C}

% --- Headings ---
\usepackage[explicit]{titlesec}
\titleformat{\chapter}[block]
  {\color{navy}\sffamily\Huge\bfseries}{Chapter \thechapter.}{0.5em}{#1}
\titleformat{\section}[block]
  {\color{navy}\sffamily\Large\bfseries}{\thesection}{0.6em}{#1}
\titleformat{\subsection}[block]
  {\color{navy}\sffamily\large\bfseries}{\thesubsection}{0.5em}{#1}

\usepackage{titletoc}
% Add a wider gap between subsection numbers and their titles in the TOC
% (otherwise "14.10" and titles starting with a digit run together).
\titlecontents{subsection}
  [4em]
  {\vspace{1pt}}
  {\contentslabel[\thecontentslabel]{3em}}
  {\hspace{-3em}}
  {\titlerule*[6pt]{.}\contentspage}

% --- Hyperlinks and bookmarks ---
\usepackage[colorlinks=true,linkcolor=navy,urlcolor=rust,citecolor=navy,
            bookmarks=true,bookmarksopen=true]{hyperref}

% --- Code listings ---
\usepackage{listings}
\lstdefinelanguage{pythonex}{
  language=Python,
  morekeywords={None,True,False,self,np,pd,plt,sklearn,xgb,lgb},
  sensitive=true,
}
\lstdefinestyle{codeblock}{
  language=pythonex,
  basicstyle=\ttfamily\footnotesize,
  keywordstyle=\color{kwd}\bfseries,
  commentstyle=\color{com}\itshape,
  stringstyle=\color{str},
  numberstyle=\color{grey}\tiny,
  numbers=left,
  numbersep=8pt,
  frame=single,
  rulecolor=\color{codeframe},
  framerule=0.4pt,
  backgroundcolor=\color{codebg},
  breaklines=true,
  breakatwhitespace=true,
  showstringspaces=false,
  columns=fullflexible,
  keepspaces=true,
  tabsize=2,
  xleftmargin=1.6em,
  xrightmargin=0.4em,
  aboveskip=12pt,
  belowskip=12pt,
}
\lstdefinestyle{htmlblock}{
  style=codeblock,
  language=HTML,
  keywordstyle=\color{rust}\bfseries,
}
\lstdefinestyle{jsblock}{
  style=codeblock,
  language=Java,
  keywordstyle=\color{kwd}\bfseries,
}
\lstset{style=codeblock}

% --- Callout box ---
\usepackage{tcolorbox}
\tcbuselibrary{breakable,skins}
\newtcolorbox{callout}[1]{%
  enhanced,breakable,
  colback=soft,colframe=ochre,
  fonttitle=\bfseries\color{ochre},
  title={#1},
  boxrule=0.6pt,arc=2pt,left=12pt,right=12pt,top=8pt,bottom=8pt,
  before skip=10pt,after skip=10pt,
}

% --- Title ---
\title{\color{navy}\sffamily\Huge Code Companion \\[6pt]
  \Large\itshape A line-by-line guide to the csdh-postop-seizure-risk
  analysis code}
\author{\sffamily Niels Pacheco-Barrios \\[2pt]
  {\small\itshape Department of Neurosurgery, Beth Israel Deaconess
  Medical Center, Harvard Medical School}}
\date{\today}
"""


def latex_escape(s):
    """Light LaTeX escape — only what is required for paragraph text."""
    return (s.replace("\\", r"\textbackslash{}")
              .replace("&", r"\&")
              .replace("%", r"\%")
              .replace("$", r"\$")
              .replace("#", r"\#")
              .replace("_", r"\_")
              .replace("{", r"\{")
              .replace("}", r"\}")
              .replace("~", r"\textasciitilde{}")
              .replace("^", r"\textasciicircum{}")
              .replace("...", r"\ldots ")
              .replace("—", r"---")
              .replace("–", r"--")
              .replace("'", r"'")
              .replace(""", r"``")
              .replace(""", r"''")
              .replace("∅", r"$\emptyset$")
              .replace("β", r"$\beta$")
              .replace("α", r"$\alpha$")
              .replace("≤", r"$\leq$")
              .replace("≥", r"$\geq$")
              .replace("±", r"$\pm$"))


def listing(code, style="codeblock"):
    """Embed a code listing with the given style."""
    return ("\n\\begin{lstlisting}[style=" + style + "]\n"
            + code + "\n\\end{lstlisting}\n")


def write_tex():
    L = [PREAMBLE, r"\begin{document}", r"\maketitle"]

    L.append(r"""
\begin{callout}{How to read this document.}
Each chapter explains one part of the codebase in plain English, then
shows the actual code with line numbers and syntax highlighting, and
then unpacks what each block is doing. You do not need to be fluent in
Python to follow along; the Python primer in Chapter~1 introduces the
constructs that recur throughout. If you only have an hour, read
Chapter~1, Chapter~2 (\texttt{\_shared.py}), and Chapter~6 (Firth +
conformal). Those three pieces are the spine of the analysis.
\end{callout}
""".replace("$", r"\$"))

    L.append(r"\tableofcontents\clearpage")

    # ── Chapter 1 — Python primer ──
    L.append(r"\section{A short Python primer for clinicians}")
    L.append(latex_escape(
        "Python is a programming language often described as 'executable "
        "pseudocode' because the constructs in a Python script tend to "
        "translate cleanly into English sentences. This chapter introduces "
        "the small set of building blocks that recur throughout our analysis "
        "code."))
    L.append(r"\subsection{How a Python file is organised}")
    L.append(latex_escape(
        "A Python file (extension .py) is read from top to bottom. The "
        "topmost few lines state the purpose of the file in a triple-quoted "
        "docstring, then import the libraries the file will need."))
    L.append(listing(
        '"""Task 24 — Firth penalized + Bayesian logistic regression.\n'
        'A docstring describes what this file does and what it produces.\n'
        '"""\n'
        'import os, sys\n'
        'import numpy as np, pandas as pd\n'
        'from sklearn.linear_model import LogisticRegression\n'
        'from _shared import load_bidmc, RES, FIG, SEED'))
    L.append(latex_escape(
        "Lines starting with the word import bring in code from elsewhere. "
        "import numpy as np gives us a shorthand: from this line on, np "
        "stands for the numpy library. from sklearn.linear_model import "
        "LogisticRegression pulls one specific tool out of a larger "
        "library — exactly like importing one drug class from a "
        "pharmacopoeia."))

    L.append(r"\subsection{Variables, lists, and dictionaries}")
    L.append(listing(
        '# A single value\n'
        'age = 73\n'
        'prevalence = 0.073\n'
        'cohort_name = "BIDMC"\n\n'
        '# A list — an ordered collection\n'
        'features = ["age", "sex", "preop_gcs", "sdh_thickness"]\n'
        'features.append("mid_shift")\n\n'
        '# A dictionary — a mapping from keys to values\n'
        'feature_means = {"age": 73, "sex": 0.68, "preop_gcs": 14.1}\n'
        'feature_means["epilepsy_hx"] = 0.02'))
    L.append(latex_escape(
        "Variables are labels that point to values. Lists hold ordered "
        "collections (square brackets). Dictionaries map a key to a value "
        "(curly braces). The hash sign # introduces a comment."))

    L.append(r"\subsection{Functions: code with a name and inputs}")
    L.append(listing(
        'def bootstrap_auc(y, p, n_boot=1000, seed=42):\n'
        '    """Compute the AUC and a 95% bootstrap confidence interval."""\n'
        '    rng = np.random.default_rng(seed)\n'
        '    bs = []\n'
        '    for _ in range(n_boot):\n'
        '        idx = rng.integers(0, len(y), len(y))\n'
        '        bs.append(roc_auc_score(y[idx], p[idx]))\n'
        '    lo, hi = np.percentile(bs, [2.5, 97.5])\n'
        '    return float(roc_auc_score(y, p)), float(lo), float(hi)'))
    L.append(latex_escape(
        "A function is a named recipe. def starts the definition; the name "
        "follows; the inputs go inside the parentheses. Inputs can have "
        "default values. The indented block under def is the body. The for "
        "line is a loop. The return statement at the end is what the "
        "function gives back to its caller."))

    L.append(r"\subsection{Pandas, numpy, and scikit-learn}")
    L.append(latex_escape(
        "Three external libraries do almost all of the analytical work in "
        "this codebase. numpy provides fast numerical arrays. pandas "
        "provides DataFrames (tables with named columns and labelled rows). "
        "scikit-learn (imported as sklearn) provides the machine-learning "
        "models, cross-validation splitters, and the standardisation "
        "machinery."))

    L.append(r"\subsection{Pipelines and the scikit-learn fit / predict pattern}")
    L.append(listing(
        'pipe = Pipeline([\n'
        '    ("imputer", SimpleImputer(strategy="median")),\n'
        '    ("scaler",  StandardScaler()),\n'
        '    ("classifier", FirthLogisticRegression()),\n'
        '])\n'
        'pipe.fit(X_train, y_train)\n'
        'probs = pipe.predict_proba(X_test)[:, 1]'))
    L.append(latex_escape(
        "A scikit-learn pipeline is a chain of steps. When we call fit, the "
        "pipeline runs imputation on the training data, then "
        "standardisation, then trains the classifier. When we call "
        "predict_proba, the pipeline runs the same imputation and "
        "standardisation (using the training-time parameters) and then "
        "asks the classifier for predicted probabilities."))
    L.append(r"\begin{callout}{Why pipelines matter.}"
              "A pipeline guarantees that the same preprocessing applied "
              "to the training data is applied — using the training-time "
              "statistics — to any new data. Without a pipeline it is "
              "easy to compute the column means on the test set and "
              "introduce a leakage bias.\\end{callout}")

    L.append(r"\subsection{Cross-validation in three lines}")
    L.append(listing(
        'rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)\n'
        'for train_idx, test_idx in rskf.split(X, y):\n'
        '    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])\n'
        '    p_oof[test_idx] = pipe.predict_proba(X.iloc[test_idx])[:, 1]'))
    L.append(latex_escape(
        "RepeatedStratifiedKFold yields pairs of (train_idx, test_idx). "
        "Each pair is one fold: train on 80% of the data, evaluate on the "
        "held-out 20%, save the predicted probabilities for the held-out "
        "patients into p_oof (out-of-fold). After all folds are processed, "
        "p_oof contains one prediction for every patient."))

    L.append(r"\clearpage")

    # ── Chapter 2 — _shared.py ──
    L.append(r"\section{\texttt{\_shared.py} — the foundations every script depends on}")
    L.append(latex_escape(
        "_shared.py is the file every other script in the project imports. "
        "It defines path constants, feature lists, loader functions, and "
        "pipeline factories. Reading this file is the single highest-yield "
        "investment of your time, because the patterns it establishes "
        "appear in every other script."))

    L.append(r"\subsection{Imports and global thread caps}")
    L.append(latex_escape(
        "The file opens with environment-variable settings that force the "
        "underlying numerical libraries (OpenBLAS, MKL, OpenMP) to use a "
        "single CPU thread per process. On Apple Silicon, multi-threaded "
        "scikit-learn can deadlock; setting n_jobs=1 throughout the "
        "codebase prevents that."))
    L.append(listing(code_snippet(SRC_SHARED, max_lines=23)))

    L.append(r"\subsection{Path constants}")
    L.append(latex_escape(
        "ROOT is the project's top-level folder. OUT is revision_analyses/, "
        "RES is its results/ subfolder, FIG is figures/, CACHE is for "
        "intermediate arrays. The for loop creates each folder if missing."))

    L.append(r"\subsection{Feature lists}")
    L.append(latex_escape(
        "POSTOP_A_FEATURES is a list of 21 column names. Defining it here, "
        "once, means every script that fits postop_A uses exactly the same "
        "21 predictors. POSTOP_B_FEATURES is the same list with the three "
        "leakage-suspect variables removed — the temporal-leakage "
        "robustness analysis."))
    L.append(listing(code_snippet(SRC_SHARED, start_token="POSTOP_A_FEATURES",
                                    max_lines=24)))

    L.append(r"\subsection{The loader functions}")
    L.append(latex_escape(
        "load_bidmc reads the cleaned BIDMC CSV and applies a single "
        "recoding. load_eicu does the same for eICU. load_eicu_pure adds "
        "an extra filter for the pure post-craniotomy sensitivity cohort."))
    L.append(listing(code_snippet(SRC_SHARED, start_token="def load_bidmc",
                                    max_lines=40)))

    L.append(r"\subsection{Pipeline factories}")
    L.append(latex_escape(
        "Each pipeline factory returns a fresh, untrained scikit-learn "
        "Pipeline. Returning a fresh pipeline matters because pipelines "
        "are mutable; fitting one in a loop would carry over state from "
        "the previous iteration."))
    L.append(listing(code_snippet(SRC_SHARED, start_token="def make_pipeline_postopA",
                                    end_token="\ndef make_pipeline_postopB",
                                    max_lines=20)))

    L.append(r"\subsection{Out-of-fold prediction helper}")
    L.append(listing(code_snippet(SRC_SHARED, start_token="def oof_predictions",
                                    end_token="\n# ─", max_lines=16)))
    L.append(latex_escape(
        "This function returns one predicted probability per patient by "
        "running repeated stratified K-fold cross-validation and averaging "
        "the predictions whenever a patient appears in multiple test folds. "
        "The accumulator trick — adding into p_acc and incrementing n_acc — "
        "is the standard idiom for averaging without storing every "
        "individual prediction."))

    L.append(r"\clearpage")

    # ── Chapter 3 — 02_calibration.py ──
    L.append(r"\section{\texttt{02\_calibration.py} — measuring how trustworthy the probabilities are}")
    L.append(latex_escape(
        "AUC measures rank-order discrimination — does the model assign "
        "higher probabilities to patients who go on to seize? Calibration "
        "measures whether those probabilities are themselves trustworthy — "
        "when the model says 10%, do roughly 10% of those patients actually "
        "seize? Calibration is the metric that shapes clinical decisions."))
    L.append(listing(code_snippet(SRC_CAL, max_lines=80)))
    L.append(latex_escape(
        "The metrics are computed by calibration_metrics in _shared.py: "
        "Brier score, calibration-in-the-large, calibration slope, "
        "intercept, ECE, MCE, and the Hosmer-Lemeshow chi-square. A "
        "well-calibrated model has Brier near p×(1−p), CITL near zero, "
        "slope near one, and intercept near zero."))

    L.append(r"\clearpage")

    # ── Chapter 4 — 04_loho.py ──
    L.append(r"\section{\texttt{04\_loho.py} — leave-one-hospital-out validation}")
    L.append(latex_escape(
        "External validation in eICU is more demanding than ordinary "
        "cross-validation because it asks: does the model generalise from "
        "hospital A to hospital B? We answer that by leaving one hospital "
        "out at a time, training on the remaining 138, and scoring the "
        "held-out hospital."))
    L.append(listing(code_snippet(SRC_LOHO, max_lines=120)))
    L.append(latex_escape(
        "loho_for handles the leave-one-hospital-out loop and writes a CSV "
        "row per hospital. random_effects_pool turns those per-hospital "
        "AUCs into a single meta-analytic estimate using the "
        "DerSimonian-Laird method."))

    L.append(r"\clearpage")

    # ── Chapter 5 — 14_decision_tree.py ──
    L.append(r"\section{\texttt{14\_decision\_tree.py} — the TreeAge-style cost-effectiveness model}")
    L.append(latex_escape(
        "Decision-tree models in health economics compute the expected "
        "cost and expected QALYs of a strategy by enumerating every "
        "possible patient trajectory, weighting each by its probability, "
        "and adding the results. This script implements that for our four "
        "strategies and renders Figure 4."))
    L.append(listing(code_snippet(SRC_TREE, max_lines=120)))
    L.append(latex_escape(
        "rollback_strategy returns the list of terminal leaves. compute_ev "
        "does the rollback: probability times payoff, summed. render_tree "
        "draws the tree in matplotlib using the TreeAge visual convention."))

    L.append(r"\clearpage")

    # ── Chapter 6 — 24_firth_bayes_lr.py ──
    L.append(r"\section{\texttt{24\_firth\_bayes\_lr.py} — the deployment model}")
    L.append(latex_escape(
        "With 48 events in BIDMC, ordinary maximum-likelihood logistic "
        "regression is unstable. Firth's penalty stabilises the coefficient "
        "estimates and produces valid confidence intervals even with rare "
        "events."))
    L.append(listing(code_snippet(SRC_FIRTH, max_lines=160)))
    L.append(latex_escape(
        "BayesianLogReg is a from-scratch implementation of logistic "
        "regression with Gaussian priors centred at user-supplied means. "
        "derive_eicu_priors fits an elastic-net LR on eICU's shared "
        "features and returns those coefficients as the prior means."))

    L.append(r"\begin{callout}{Headline finding}"
              "Firth matches the BalancedRandomForest baseline on AUC "
              "(0.681 vs 0.676; DeLong $p$ = 0.81) with a 3.3-fold improvement "
              "in Brier score. The Bayesian-with-eICU-priors variant "
              "degrades discrimination significantly — a biological-mismatch "
              "warning, not a statistical failure.\\end{callout}")

    L.append(r"\clearpage")

    # ── Chapter 7 — 25_conformal_prediction.py ──
    L.append(r"\section{\texttt{25\_conformal\_prediction.py} — turning probabilities into bedside decisions}")
    L.append(latex_escape(
        "Conformal prediction turns any classifier into a procedure that "
        "emits prediction sets with distribution-free coverage guarantees. "
        "The class-conditional variant (Mondrian) guarantees coverage "
        "separately for each true class."))
    L.append(listing(code_snippet(SRC_CONF, max_lines=140)))
    L.append(latex_escape(
        "class_conditional_conformal computes the nonconformity score "
        "1 − P(true class) on every calibration example, takes the "
        "(1−α)(n+1)/n empirical quantile separately for the positive and "
        "negative classes, and for each test point includes each class in "
        "the prediction set iff its nonconformity does not exceed that "
        "class's quantile."))

    L.append(r"\clearpage")

    # ── Chapter 8 — 16_voi_evpi.py ──
    L.append(r"\section{\texttt{16\_voi\_evpi.py} — value-of-information analysis}")
    L.append(latex_escape(
        "Value-of-information analysis answers: of all the uncertain "
        "parameters in our decision model, which would actually change "
        "the optimal strategy if we knew them perfectly? EVPI is the "
        "population-scale upper bound on the value of any future study. "
        "EVPPI ranks per-parameter research priorities."))
    L.append(listing(code_snippet(SRC_VOI, max_lines=140)))
    L.append(latex_escape(
        "compute_evpi is the population-EVPI formula. "
        "evppi_strong_oakley implements the Strong-Oakley non-parametric "
        "regression approximation per parameter and feeds the tornado "
        "chart in Figure 6."))

    L.append(r"\clearpage")

    # ── Chapter 9 — 29_main_figures_jnnp.py ──
    L.append(r"\section{\texttt{29\_main\_figures\_jnnp.py} — publication-grade figures}")
    L.append(latex_escape(
        "Every main paper figure is built by a dedicated function "
        "(figure_1 ... figure_6) with a consistent house style: Helvetica "
        "sans-serif, open-box axes, bold uppercase panel labels, a "
        "restrained CMYK-safe palette, 300-dpi PNG and editable vector "
        "PDF output."))
    L.append(listing(code_snippet(SRC_FIG, max_lines=110)))
    L.append(latex_escape(
        "The plt.rcParams.update block sets the house style once; every "
        "panel inherits from it. add_panel_label puts the bold A/B/C "
        "labels; style_axis applies the tick direction; "
        "figure_legend_below places a single shared legend beneath the "
        "figure so legends never overlap data."))

    L.append(r"\clearpage")

    # ── Chapter 10 — 27_build_jnnp_manuscript.py ──
    L.append(r"\section{\texttt{27\_build\_jnnp\_manuscript.py} — building the Word manuscript from code}")
    L.append(latex_escape(
        "This script writes the manuscript .docx file end-to-end using "
        "python-docx. Building the manuscript programmatically means every "
        "numerical claim in the text is traceable to the results CSV that "
        "produced it, and every version of the manuscript is a "
        "deterministic function of the underlying analysis."))
    L.append(listing(code_snippet(SRC_MANUS, max_lines=80)))
    L.append(latex_escape(
        "The structure mirrors a normal Word workflow: setup_document sets "
        "page margins; add_heading, add_para, add_runs handle text; "
        "register_figure and register_table accumulate figures and tables; "
        "render_collected_tables_and_figures emits them all in a single "
        "Tables-and-Figures section at the end."))

    L.append(r"\clearpage")

    # ── Chapter 11 — 30_export_calculator_assets.py ──
    L.append(r"\section{\texttt{30\_export\_calculator\_assets.py} — packaging the model for the dashboard}")
    L.append(latex_escape(
        "The browser-side calculator cannot run scikit-learn — it is "
        "JavaScript. To let the browser compute the same prediction the "
        "Python model would, this script extracts the deployment-ready "
        "coefficients, the feature scaler statistics, and the conformal "
        "quantiles and writes them to site/model_assets.json."))
    L.append(listing(code_snippet(SRC_EXPORT, max_lines=120)))
    L.append(latex_escape(
        "The Firth model is fit on the full BIDMC cohort. The "
        "class-conditional conformal quantiles are computed by repeated "
        "5-fold splits with 75/25 train/calibration partitioning; the "
        "(1−α)(n+1)/n quantile of each class is taken. Everything is "
        "bundled into a single JSON file."))

    L.append(r"\clearpage")

    # ── Chapter 12 — Dashboards ──
    L.append(r"\section{The interactive dashboards — HTML, CSS, JavaScript}")
    L.append(latex_escape(
        "The three dashboards on the companion site are static HTML pages "
        "served by GitHub Pages. Each page bundles its own JavaScript so "
        "all computation happens in the user's browser."))

    L.append(r"\subsection{Anatomy of an HTML page}")
    L.append(listing(
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '  <meta charset="UTF-8">\n'
        '  <title>Risk calculator</title>\n'
        '  <style> /* CSS — visual styling */ </style>\n'
        '</head>\n'
        '<body>\n'
        '  <header>...</header>\n'
        '  <main>...</main>\n'
        '  <script> /* JavaScript — behaviour */ </script>\n'
        '</body>\n'
        '</html>', style="htmlblock"))
    L.append(latex_escape(
        "HTML defines structure; CSS defines visual style; JavaScript "
        "defines behaviour."))

    L.append(r"\subsection{How the risk calculator works}")
    L.append(latex_escape(
        "On page load the calculator fetches model_assets.json, extracts "
        "the Firth coefficients and the conformal quantiles, and attaches "
        "an event listener to every form input. Whenever the user changes "
        "any input, the recompute function fires: it reads all 21 feature "
        "values, z-scales them using the means and SDs from the JSON, "
        "computes the dot product with the coefficients, adds the "
        "intercept, applies the sigmoid, and produces a probability."))
    # Extract recompute function from calculator.html
    js_start = SRC_CALC.find("function recompute")
    js_end = SRC_CALC.find("\ndocument.addEventListener", js_start)
    if js_start >= 0 and js_end > js_start:
        L.append(listing(code_snippet(SRC_CALC[js_start:js_end].rstrip(),
                                        max_lines=55), style="jsblock"))
    L.append(latex_escape(
        "The dot product is a single for loop. Each iteration takes one "
        "raw feature value, z-scales it, multiplies by the corresponding "
        "Firth coefficient, and accumulates into z. After the loop, the "
        "intercept is added and the sigmoid applied: p = 1/(1+exp(-z)). "
        "That gives the predicted probability. The conformal logic that "
        "follows checks whether 1-p is below each class-conditional "
        "quantile; the answer determines the prediction set and the "
        "recommendation."))

    L.append(r"\subsection{How the savings calculator works}")
    L.append(latex_escape(
        "The savings calculator's JavaScript multiplies the per-patient "
        "ΔCost and ΔQALY values by the user-specified cohort size, "
        "applies the chosen discount rate over the chosen horizon, and "
        "presents the result as KPI cards and a net-monetary-benefit bar "
        "chart. No external libraries are used — the bars are styled "
        "<div> elements whose width is set proportionally."))

    L.append(r"\subsection{How the interactive callgraph works}")
    L.append(latex_escape(
        "The callgraph dashboard uses one external library — vis-network "
        "— loaded from a CDN. On page load the script fetches "
        "callgraph.json, maps each script to a node with category-coloured "
        "fill, attaches the per-node function inventory as a hidden "
        "property, and renders the network with a force-directed layout."))

    L.append(r"\clearpage")

    # ── Chapter 13 — Callgraph ──
    L.append(r"\section{The callgraph — how the scripts depend on each other}")
    L.append(latex_escape(
        "Every analysis script in the repository imports _shared.py. Many "
        "also import each other. The callgraph is generated by parsing "
        "each .py file's Abstract Syntax Tree (Python's machine-readable "
        "representation of its own source) and extracting the import "
        "statements and top-level function definitions."))
    L.append(r"\begin{callout}{How to use the callgraph when you change something.}"
              "Before you modify any function in "
              r"\texttt{\_shared.py}, look at the callgraph and ask yourself: "
              "which scripts import this function? Any script with an arrow "
              r"pointing into \texttt{\_shared.py} depends on its behaviour. "
              "Test downstream before publishing the change."
              r"\end{callout}")

    L.append(r"\clearpage")

    # ── Chapter 14 — Others ──
    L.append(r"\section{Other scripts — what they do, in one paragraph each}")
    others = [
        (r"\texttt{03\_dca.py}",
         "Decision-curve analysis. Reads cached out-of-fold predictions, "
         "computes Vickers net benefit at every probability threshold from "
         "0 to 30 percent, plots the model curve against treat-all and "
         "treat-none, and writes the summary CSV used by Figure 2 panel B."),
        (r"\texttt{05\_temporal\_leakage.py}",
         "Temporal-leakage audit. Excludes the 24-hour and 48-hour rolling "
         "lab/vital features and the prophylactic-AED indicator, refits, "
         "and reports the resulting AUC."),
        (r"\texttt{06\_overfitting.py}",
         "Overfitting diagnostics. Repeats the BalancedRandomForest fit "
         "with nested cross-validation, compares variable-importance ranks "
         "across folds (Spearman rho across 25 folds = 0.98), and produces "
         "a learning-curve plot."),
        (r"\texttt{07\_missing\_data.py}",
         "Missing-data sensitivity. Runs Little's MCAR test, computes the "
         "missingness-versus-outcome chi-square per feature, and pools AUC "
         "estimates across ten multiply-imputed datasets via Rubin's rules."),
        (r"\texttt{08\_eicu\_cohort.py}",
         "eICU cohort definition sensitivity. Fits the model in each of "
         "four prespecified cohort strata and reports bootstrap 95% AUC CIs."),
        (r"\texttt{09\_competing\_risks.py}",
         "Cause-specific Cox model plus IPCW Fine-Gray subdistribution-"
         "hazard model. Reports concordance indices and the "
         "Grambsch-Therneau Schoenfeld test."),
        (r"\texttt{10\_11\_cea\_pairwise.py}",
         "Cost-effectiveness analysis with 10,000-iteration PSA. Defines "
         "the Params dataclass; samples each parameter from its own beta "
         "or gamma distribution; runs each of the four strategies; reports "
         "the cost-effectiveness plane and the acceptability curve."),
        (r"\texttt{12\_nis\_seizure\_reclassify.py}",
         "The NIS outcome-correction analysis. Distinguishes acute "
         "symptomatic seizure from pre-existing epilepsy and reports both "
         "definitions' AUC."),
        (r"\texttt{13\_nis\_grouped\_lasso.py}",
         "Group-LASSO with lambda-path tuning on the NIS chronic+surgical "
         "cohort under the corrected outcome."),
        (r"\texttt{15\_radiology\_nlp.py}",
         "Regex-pattern radiology NLP extractor with a synthetic "
         "validation set. Released as a tool; not promoted to a paper "
         "contribution."),
        (r"\texttt{17\_build\_slides.py}",
         "Builds the 15-minute oral-presentation deck using python-pptx."),
        (r"\texttt{18\_bidmc\_optimize.py}",
         "BIDMC model-optimization sweep. Optuna for 40 trials each on "
         "XGBoost and LightGBM; stacking ensemble; DeLong tests against "
         "the baseline."),
        (r"\texttt{19\_transfer\_learning.py}",
         "eICU to BIDMC transfer-learning experiment. Trains an XGBoost on "
         "the seven shared features, scores BIDMC patients, and uses the "
         "score as an additional feature in the BIDMC model."),
        (r"\texttt{20\_build\_manuscript.py}",
         "An earlier manuscript-build script. Superseded by 27."),
        (r"\texttt{21\_imbalance\_sweep.py}",
         "Eleven-method class-imbalance sweep on BIDMC."),
        (r"\texttt{22\_diverse\_stacking.py}",
         "Diverse-base stacking ensemble (LR + BalancedRF + XGBoost + KNN "
         "+ RBF-SVM with a logistic meta-learner)."),
        (r"\texttt{23\_tabpfn\_eval.py}",
         "Attempted TabPFN v2 evaluation. Reports the API-credential "
         "requirement and falls back."),
        (r"\texttt{26\_main\_figures.py}",
         "Earlier figure-builder that composes existing PNGs via PIL. "
         "Superseded by 29."),
        (r"\texttt{28\_make\_callgraph.py}",
         "Parses every .py file via ast, emits Markdown + Mermaid callgraph."),
        (r"\texttt{31\_export\_callgraph\_json.py}",
         "Same parsing pipeline as 28, emits JSON for the interactive "
         "callgraph dashboard."),
        (r"\texttt{32\_build\_code\_companion.py}",
         "Builds the Word version of this document."),
        (r"\texttt{33\_build\_code\_companion\_pdf.py}",
         "This script — the one that built the PDF you are reading."),
    ]
    for name, blurb in others:
        L.append(r"\subsection{" + name + r"}")
        L.append(latex_escape(blurb))

    L.append(r"\clearpage")

    # ── Chapter 15 — Skills ──
    L.append(r"\section{Further study — Claude skills that help you learn the code}")
    L.append(latex_escape(
        "Claude ships with a library of task-focused expert prompts called "
        "skills, each invoked by typing the skill name with a leading "
        "slash. The following skills are particularly relevant to studying "
        "this codebase."))
    skill_recs = [
        (r"\texttt{/claude-code-guide}",
         "General Python idioms, scikit-learn patterns, the Anthropic API."),
        (r"\texttt{/code-review-excellence}",
         "Structured second-pair-of-eyes review with prioritised feedback."),
        (r"\texttt{/documentation-writer}",
         "Docstrings, README sections, function-level comments (Diataxis)."),
        (r"\texttt{/python-project-structure}",
         "Why a Python project is organised the way it is."),
        (r"\texttt{/scientific-writing}",
         "Translating analysis results into manuscript prose."),
        (r"\texttt{/scientific-critical-thinking}",
         "Interrogating methodological choices and bias diagnostics."),
        (r"\texttt{/visualization-best-practices}",
         "Refining figures, colour-blind safety, annotation placement."),
        (r"\texttt{/statistical-analysis}",
         "Choosing statistical tests, assumption checking, reporting."),
        (r"\texttt{/the-humanizer}",
         "Removing AI-pattern phrasing from a draft."),
        (r"\texttt{/peer-review}",
         "Structured manuscript review applying CONSORT or STROBE or TRIPOD."),
        (r"\texttt{/literature-review}",
         "Comprehensive systematically-searched literature review."),
        (r"\texttt{/exploratory-data-analysis}",
         "Initial inspection of a new dataset with quality metrics."),
        (r"\texttt{/scikit-learn}",
         "Specific questions about scikit-learn pipelines and CV."),
        (r"\texttt{/statsmodels}",
         "Formal frequentist inference beyond scikit-learn."),
        (r"\texttt{/pymc}",
         "Fitting Bayesian models with MCMC."),
        (r"\texttt{/improve-codebase-architecture}",
         "Refactoring after reading; reducing duplication."),
    ]
    for name, body in skill_recs:
        L.append(r"\subsection*{" + name + r"}")
        L.append(latex_escape(body))

    L.append(r"\begin{callout}{A short reading order for first-time exploration.}"
              r"(1) Read Chapter 1 of this document to learn the Python "
              r"constructs you will see. (2) Read \texttt{\_shared.py} with "
              r"Chapter 2 of this document open beside you. "
              r"(3) Read \texttt{02\_calibration.py} and "
              r"\texttt{24\_firth\_bayes\_lr.py} with Chapters 3 and 6 open. "
              r"(4) Open the calculator dashboard in your browser, then come "
              r"back to Chapter 12 to see how the JavaScript implements the "
              r"same model. By that point the rest of the repository will "
              r"feel familiar.\end{callout}")

    L.append(r"\end{document}")

    TEX_PATH.write_text("\n".join(L), encoding="utf-8")
    print(f"[OK] wrote {TEX_PATH} ({os.path.getsize(TEX_PATH)/1024:.1f} KB)")


def compile_pdf():
    """Compile the LaTeX file to PDF using xelatex (twice for TOC).
    Treats non-zero exit codes as warnings rather than errors if a PDF was
    still produced (xelatex returns non-zero for typesetting badness
    even when the output PDF is complete and usable)."""
    cwd = DOCS
    pdf = cwd / "Code_Companion.pdf"
    last_proc = None
    for run in (1, 2):
        last_proc = subprocess.run(
            ["xelatex", "-interaction=nonstopmode",
             "Code_Companion.tex"],
            cwd=cwd, capture_output=True, text=True
        )
    # Success = PDF exists; report any warnings the user might care about
    if pdf.exists():
        if last_proc.returncode != 0:
            print("[note] xelatex returned non-zero exit code but a PDF was "
                  "produced. Inspect Code_Companion.log for warnings.")
        for ext in (".aux", ".out", ".toc"):
            f = cwd / f"Code_Companion{ext}"
            if f.exists(): f.unlink()
        return True
    print("[xelatex] no PDF produced")
    print((last_proc.stdout or "")[-2000:])
    return False


def main():
    write_tex()
    if compile_pdf():
        if PDF_PATH.exists():
            print(f"[OK] {PDF_PATH} ({os.path.getsize(PDF_PATH)/1024:.1f} KB)")
        else:
            print("[ERROR] xelatex returned 0 but no PDF was produced")
    else:
        print("[WARN] PDF compile failed — see log above. The .tex file is "
              "still available for inspection.")


if __name__ == "__main__":
    main()
