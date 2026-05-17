.PHONY: help check test fmt lint doc ci ci-fast hooks

help:
	@echo "Available targets:"
	@echo "  check    Check workspace compiles"
	@echo "  test     Run all tests (sherpa-onnx feature)"
	@echo "  fmt      Format code"
	@echo "  lint     Run clippy with warnings as errors"
	@echo "  doc      Build and open docs in browser"
	@echo "  ci       Run all CI checks locally"
	@echo "  ci-fast  Quick pre-push subset (fmt + clippy + test)"
	@echo "  hooks    Install .githooks/ (pre-commit: fmt; pre-push: ci-fast)"

check:
	cargo check --workspace --all-features

test:
	cargo test --workspace --features sherpa-onnx

fmt:
	cargo fmt --all

lint:
	cargo clippy --workspace --all-features -- -D warnings

doc:
	cargo doc --no-deps -p wavekat-asr --all-features --open

ci:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-features -- -D warnings
	cargo test -p wavekat-asr --no-default-features
	cargo test -p wavekat-asr --no-default-features --features sherpa-onnx
	cargo doc --no-deps -p wavekat-asr --all-features

ci-fast:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-features -- -D warnings
	cargo test -p wavekat-asr --no-default-features --features sherpa-onnx

hooks:
	git config core.hooksPath .githooks
	@echo "Installed git hooks from .githooks/ (bypass with --no-verify)."
