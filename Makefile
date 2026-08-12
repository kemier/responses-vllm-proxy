# Build / install / run helpers for vllm-proxy.
#
#   make            build release binary (in-tree)
#   make install    rebuild + install to $(PREFIX) (does NOT start/restart the server)
#   make uninstall  remove the installed copy (does not stop a running proxy)
#   make clean      cargo clean
#
# Override install location:
#   make install PREFIX=/opt/vllm-proxy

TARGET   := target/release/vllm-proxy
PREFIX   ?= /workspace/git/ezennin/.local/share/vllm-proxy
BINDIR   ?= /workspace/git/ezennin/.local/bin
SRC      := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

.PHONY: all build install uninstall clean

all: build

build: $(TARGET)

$(TARGET):
	cargo build --release

# Install the freshly built binary + relocatable scripts + config into PREFIX,
# drop a wrapper on BINDIR/vllm-proxy (on PATH). Does NOT start the server.
install: build
	@echo "==> Installing to $(PREFIX)"
	@mkdir -p $(PREFIX)/bin $(PREFIX)/scripts $(PREFIX)/logs
	@install -m 0755 $(TARGET) $(PREFIX)/bin/vllm-proxy
	@install -m 0755 $(SRC)/start.sh $(PREFIX)/start.sh
	@install -m 0755 $(SRC)/scripts/update_catalog.sh $(PREFIX)/scripts/update_catalog.sh
	@install -m 0755 $(SRC)/scripts/check_health.sh $(PREFIX)/scripts/check_health.sh
	@install -m 0644 $(SRC)/config.yaml $(PREFIX)/config.yaml
	@printf '#!/bin/bash
exec $(PREFIX)/start.sh "$$@"
' > $(BINDIR)/vllm-proxy
	@chmod 0755 $(BINDIR)/vllm-proxy
	@PIDFILE="/tmp/vllm-proxy-$$USER.pid"; \
	 if [ -f "$$PIDFILE" ]; then \
	   OLD_PID=$$(cat "$$PIDFILE" 2>/dev/null); \
	   if [ -n "$$OLD_PID" ] && kill -0 "$$OLD_PID" 2>/dev/null; then \
	     echo "ERROR: vllm-proxy is already running (PID $$OLD_PID). Stop it first with: kill <PID>" >&2; \
	     exit 1; \
	   fi; \
	 fi; \
	 echo "==> Done"

uninstall:
	@rm -rf $(PREFIX)
	@rm -f $(BINDIR)/vllm-proxy
	@echo "removed $(PREFIX) and $(BINDIR)/vllm-proxy"

clean:
	cargo clean