CONFIG ?= work_dirs/bcss/classification/config.yaml
GPU ?= 0
LOG_DIR ?= logs

.PHONY: train
train:
	@STAMP=$$(date +"%Y%m%d-%H%M%S"); \
	mkdir -p $(LOG_DIR); \
	LOG=$(LOG_DIR)/train-$${STAMP}.log; \
	echo ">>> Logging to $$LOG"; \
	python main.py --config $(CONFIG) --gpu $(GPU) 2>&1 | tee $$LOG

.PHONY: help
help:
	@echo "Makefile commands:"
	@echo "  make train      Train the model with the specified configuration."
	@echo "                  Optional variables:"
	@echo "                    CONFIG - Path to the config file (default: work_dirs/bcss/classification/config.yaml)"
	@echo "                    GPU    - GPU id to use (default: 0)"
	@echo "                    LOG_DIR - Directory to save logs (default: logs)"
	@echo "  make help       Show this help message."

