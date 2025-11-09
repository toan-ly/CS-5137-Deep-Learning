
CHECKPOINTS_DIR := checkpoints
PLOTS_DIR := results/predictions
ZIP_FILE := checkpoints.zip
DIR ?= unet    

unzip:
	@echo ">>> Checking for $(ZIP_FILE) ..."
	@if [ ! -f $(ZIP_FILE) ]; then \
		echo "Error: $(ZIP_FILE) not found!"; \
		exit 1; \
	fi

	@echo ">>> Unzipping $(ZIP_FILE) ..."
	@unzip -o $(ZIP_FILE) -d temp > /dev/null

	@mkdir -p $(CHECKPOINTS_DIR)/$(DIR)
	@mkdir -p $(PLOTS_DIR)/$(DIR)

	@echo ">>> Moving weights to $(CHECKPOINTS_DIR)/$(DIR)/ ..."
	@mv temp/checkpoints/* $(CHECKPOINTS_DIR)/$(DIR)/
	@echo ">>> Moving plots to $(PLOTS_DIR)/$(DIR)/ ..."
	@mv temp/* $(PLOTS_DIR)/$(DIR)/

	@rm -f $(ZIP_FILE)
	@rm -rf temp
	@rm -rf ${PLOTS_DIR}/${DIR}/checkpoints

	@echo ">>> Done! Files saved to:"
	@echo "    - $(CHECKPOINTS_DIR)/$(DIR)/"
	@echo "    - $(PLOTS_DIR)/$(DIR)/"

test:
	uv run -m src.main