ENV_NAME := chowder
ENV_FILES_DIR := conda
CORE_DEPS := $(ENV_FILES_DIR)/environment.yml
DEV_DEPS := $(ENV_FILES_DIR)/environment-dev.yml

.PHONY: _install_core _update_core _update_dev _create_env _install_mamba

# Create an environment with core and development dependencies
install: install_core _update_dev _install_package_editable

# Create an environment with the core dependencies
install_core: _install_core 

# Update the complete development environment from both environment.yml and environment-dev.yml
update: _update_core _update_dev

# Update the core development dependencies from environment.yml
update_core: _update_core

test:
	coverage run -m pytest tests/unit tests/integration
	coverage report
# Install core dependencies from the conda/environment.yml file
_install_core:
	mamba env create -n $(ENV_NAME) -f $(CORE_DEPS)

# Update core dependencies from the conda/environment.yml file
_update_core:
	mamba env update -n $(ENV_NAME) -f $(CORE_DEPS)

# Update development dependencies from the conda/environment-dev.yml file
_update_dev:
	mamba env update -n $(ENV_NAME) -f $(DEV_DEPS)

# Install the package itself in editable mode
_install_package_editable:
	mamba run -n $(ENV_NAME) pip install --no-deps -e .

# Train the model
train:
	python chowder/train.py

# Predict the test data
predict:
	python chowder/predict.py