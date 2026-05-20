# Makefile for csdh-postop-seizure-risk
#
# Common reproduction targets.  All scripts run with n_jobs=1 and SEED=42;
# rebuilding from a clean clone reproduces the manuscript's exact numbers.

PY := python
SCRIPTS := scripts

.PHONY: help all calibration loho models conformal cea voi figures \
        manuscript dashboard companion clean

help:
	@echo "Targets:"
	@echo "  make calibration   — calibration metrics + curves (script 02)"
	@echo "  make loho          — leave-one-hospital-out + random-effects pool (script 04)"
	@echo "  make models        — Firth + Bayesian + diverse stacking + imbalance sweep"
	@echo "                       (scripts 21, 22, 24, 25)"
	@echo "  make conformal     — class-conditional conformal sets (script 25)"
	@echo "  make cea           — base-case CEA + PSA + decision tree (scripts 10, 14)"
	@echo "  make voi           — value-of-information (script 16)"
	@echo "  make figures       — main paper figures F1-F6 (script 29) + graphical abstract (34)"
	@echo "  make manuscript    — Word .docx (script 27)"
	@echo "  make companion     — Code Companion PDF + DOCX (scripts 32, 33)"
	@echo "  make dashboard     — model_assets.json + callgraph.json for the interactive site"
	@echo "                       (scripts 30, 31)"
	@echo "  make all           — everything"
	@echo "  make clean         — remove cache/ and intermediate logs"

calibration:
	$(PY) $(SCRIPTS)/02_calibration.py

loho:
	$(PY) $(SCRIPTS)/04_loho.py

models:
	$(PY) $(SCRIPTS)/21_imbalance_sweep.py
	$(PY) $(SCRIPTS)/22_diverse_stacking.py
	$(PY) $(SCRIPTS)/24_firth_bayes_lr.py
	$(PY) $(SCRIPTS)/25_conformal_prediction.py

conformal:
	$(PY) $(SCRIPTS)/25_conformal_prediction.py

cea:
	$(PY) $(SCRIPTS)/10_11_cea_pairwise.py
	$(PY) $(SCRIPTS)/14_decision_tree.py

voi:
	$(PY) $(SCRIPTS)/16_voi_evpi.py

figures:
	$(PY) $(SCRIPTS)/29_main_figures_jnnp.py
	$(PY) $(SCRIPTS)/34_graphical_abstract.py

manuscript:
	$(PY) $(SCRIPTS)/27_build_jnnp_manuscript.py

companion:
	$(PY) $(SCRIPTS)/32_build_code_companion.py
	$(PY) $(SCRIPTS)/33_build_code_companion_pdf.py

dashboard:
	$(PY) $(SCRIPTS)/30_export_calculator_assets.py
	$(PY) $(SCRIPTS)/31_export_callgraph_json.py

all: calibration loho models cea voi figures dashboard manuscript companion

clean:
	rm -rf cache/ *.log
