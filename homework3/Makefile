
CHECKPOINTS_DIR := checkpoints
ZIP_FILE := checkpoints.zip
DIR ?= unet    

unzip:
	@echo ">>> Unzipping $(ZIP_FILE) into $(CHECKPOINTS_DIR)/$(DIR)/ ..."
	@mkdir -p $(CHECKPOINTS_DIR)/$(DIR)
	@unzip -o $(ZIP_FILE) -d $(CHECKPOINTS_DIR)/$(DIR) > /dev/null
	@echo ">>> Done! Files saved to $(CHECKPOINTS_DIR)/$(DIR)/"
	@rm -f $(ZIP_FILE)