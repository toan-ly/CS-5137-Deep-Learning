
PYTHON := uv run python

DATA_DIR := data
WEIGHTS_DIR := weights
RESULTS_DIR := results
TEMP_DIR := .temp

eval: 
	@echo "Running evaluation..."
	$(PYTHON) src/evaluation.py

unzip: check_dirs
	@echo "Organizing training results..."
	unzip all_files.zip -d $(TEMP_DIR)
	mv -f $(TEMP_DIR)/data/processed/* $(DATA_DIR)/
	mv -f $(TEMP_DIR)/weights/* $(WEIGHTS_DIR)/
	mv -f $(TEMP_DIR)/results/histories.csv $(RESULTS_DIR)/histories.csv
	rm -rf $(TEMP_DIR)
	rm -rf all_files.zip
	@echo "✅ Done!"

check_dirs:
	@mkdir -p $(DATA_DIR) $(WEIGHTS_DIR) $(RESULTS_DIR)
