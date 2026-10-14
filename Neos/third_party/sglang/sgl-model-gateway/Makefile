                        
                                                            

                           
PYTHON_DIR := bindings/python

                                                                              
                                           
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
JOBS ?= $(shell echo $$(($(NPROC) > 16 ? 16 : $(NPROC))))

                                                                 
SCCACHE := $(shell which sccache 2>/dev/null)
ifdef SCCACHE
    export RUSTC_WRAPPER := $(SCCACHE)
    $(info Using sccache for compilation caching)
else
    $(info sccache not found. Install it for faster builds: cargo install sccache)
endif

.PHONY: help build test clean docs check fmt dev-setup pre-commit setup-sccache sccache-stats sccache-clean sccache-stop \
        python-dev python-build python-build-release python-install python-clean python-test python-check \
        show-version bump-version release-notes

help:                          
	@echo "Model Gateway Development Commands"
	@echo "=================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

build:                                     
	@echo "Building SGLang Model Gateway..."
	@cargo build --release

test:                 
	@echo "Running tests..."
	@cargo test

clean:                         
	@echo "Cleaning build artifacts..."
	@cargo clean

docs:                                   
	@echo "Generating documentation..."
	@cargo doc --open

check:                              
	@echo "Running cargo check..."
	@cargo check
	@echo "Running clippy..."
	@cargo clippy --all-targets --all-features -- -D warnings

fmt:                            
	@echo "Formatting code..."
	@rustup run nightly cargo fmt

                                
dev-setup: build test                                  
	@echo "Development environment ready!"

pre-commit: fmt check test                         
	@echo "Pre-commit checks passed!"

                            
setup-sccache:                                 
	@echo "Setting up sccache..."
	@./scripts/setup-sccache.sh

sccache-stats:                           
	@if [ -n "$(SCCACHE)" ]; then \
		echo "sccache statistics:"; \
		sccache -s; \
	else \
		echo "sccache not installed. Run 'make setup-sccache' to install it."; \
	fi

sccache-clean:                       
	@if [ -n "$(SCCACHE)" ]; then \
		echo "Clearing sccache cache..."; \
		sccache -C; \
		echo "sccache cache cleared"; \
	else \
		echo "sccache not installed"; \
	fi

sccache-stop:                           
	@if [ -n "$(SCCACHE)" ]; then \
		echo "Stopping sccache server..."; \
		sccache --stop-server || true; \
	else \
		echo "sccache not installed"; \
	fi

                                   
python-dev:                                                                 
	@echo "Building Python bindings in development mode (using $(JOBS) parallel jobs with sccache)..."
	@cd $(PYTHON_DIR) && CARGO_BUILD_JOBS=$(JOBS) maturin develop

python-build:                                                           
	@echo "Building Python wheel (release, vendored OpenSSL, using $(JOBS) parallel jobs with sccache)..."
	@cd $(PYTHON_DIR) && CARGO_BUILD_JOBS=$(JOBS) maturin build --release --out dist --features vendored-openssl

python-build-release: python-build                          

python-install: python-build                                  
	@echo "Installing Python wheel..."
	@pip install --force-reinstall $(PYTHON_DIR)/dist/*.whl
	@echo "Python package installed!"

python-clean:                                
	@echo "Cleaning Python build artifacts..."
	@rm -rf $(PYTHON_DIR)/dist/
	@rm -rf $(PYTHON_DIR)/target/
	@rm -rf $(PYTHON_DIR)/sglang_router.egg-info/
	@rm -rf $(PYTHON_DIR)/sglang_router/__pycache__/
	@find $(PYTHON_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find $(PYTHON_DIR) -name "*.pyc" -delete 2>/dev/null || true
	@echo "Python build artifacts cleaned!"

python-test:                    
	@echo "Running Python tests..."
	@pytest e2e_test/ -v

python-check:                                   
	@echo "Checking Python package..."
	@cd $(PYTHON_DIR) && CARGO_BUILD_JOBS=$(JOBS) maturin build --release --out dist --features vendored-openssl
	@pip install twine 2>/dev/null || true
	@twine check $(PYTHON_DIR)/dist/*
	@echo "Python package check passed!"

                    
dev: python-dev                                                               

install: python-install                                

                    
VERSION_FILES := Cargo.toml \
                 bindings/golang/Cargo.toml \
                 bindings/python/Cargo.toml \
                 bindings/python/pyproject.toml \
                 bindings/python/src/sglang_router/version.py

show-version:                                         
	@echo "Current versions:"
	@echo "  Cargo.toml:                 $$(grep -m1 '^version = ' Cargo.toml | sed 's/version = "\(.*\)"/\1/')"
	@echo "  bindings/golang/Cargo.toml: $$(grep -m1 '^version = ' bindings/golang/Cargo.toml | sed 's/version = "\(.*\)"/\1/')"
	@echo "  bindings/python/Cargo.toml: $$(grep -m1 '^version = ' bindings/python/Cargo.toml | sed 's/version = "\(.*\)"/\1/')"
	@echo "  bindings/python/pyproject.toml: $$(grep -m1 '^version = ' bindings/python/pyproject.toml | sed 's/version = "\(.*\)"/\1/')"
	@echo "  bindings/python/.../version.py: $$(grep '__version__' bindings/python/src/sglang_router/version.py | sed 's/__version__ = "\(.*\)"/\1/')"

bump-version:                                                                          
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make bump-version VERSION=<new-version>"; \
		echo "Example: make bump-version VERSION=0.3.3"; \
		echo ""; \
		echo "Current version:"; \
		grep -m1 '^version = ' Cargo.toml | sed 's/version = "\(.*\)"/  \1/'; \
		exit 1; \
	fi
	@echo "Bumping version to $(VERSION)..."
                                   
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' Cargo.toml && rm -f Cargo.toml.bak
                                    
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' bindings/golang/Cargo.toml && rm -f bindings/golang/Cargo.toml.bak
                                    
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' bindings/python/Cargo.toml && rm -f bindings/python/Cargo.toml.bak
                         
	@sed -i.bak 's/^version = ".*"/version = "$(VERSION)"/' bindings/python/pyproject.toml && rm -f bindings/python/pyproject.toml.bak
                     
	@sed -i.bak 's/__version__ = ".*"/__version__ = "$(VERSION)"/' bindings/python/src/sglang_router/version.py && rm -f bindings/python/src/sglang_router/version.py.bak
	@echo "Version updated to $(VERSION) in all files:"
	@echo "  - Cargo.toml"
	@echo "  - bindings/golang/Cargo.toml"
	@echo "  - bindings/python/Cargo.toml"
	@echo "  - bindings/python/pyproject.toml"
	@echo "  - bindings/python/src/sglang_router/version.py"
	@echo ""
	@echo "Verify with: make show-version"

release-notes:                                                                                                          
	@if [ -z "$(PREV)" ] || [ -z "$(CURR)" ]; then \
		echo "Usage: make release-notes PREV=<previous-tag> CURR=<current-tag>"; \
		echo "Example: make release-notes PREV=gateway-v0.2.2 CURR=gateway-v1.0.0"; \
		echo ""; \
		echo "Options:"; \
		echo "  OUTPUT=<file>     Save to file (default: stdout)"; \
		echo "  CREATE_RELEASE=1  Create GitHub draft release via gh CLI (default: draft)"; \
		echo "  DRAFT=0           Publish release immediately (skip draft)"; \
		exit 1; \
	fi
	@ARGS="$(PREV) $(CURR)"; \
	if [ -n "$(OUTPUT)" ]; then \
		ARGS="$$ARGS --output $(OUTPUT)"; \
	fi; \
	if [ "$(CREATE_RELEASE)" = "1" ]; then \
		ARGS="$$ARGS --create-release"; \
		if [ "$(DRAFT)" = "0" ]; then \
			ARGS="$$ARGS --no-draft"; \
		fi; \
	fi; \
	./scripts/generate_gateway_release_notes.sh $$ARGS
