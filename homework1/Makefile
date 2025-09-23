.PHONY: run clean

lr ?= 0.01

run:
	uv run python src/main.py --lr=$(lr)

run-all:
	$(MAKE) clean
	$(MAKE) run lr=0.1
	$(MAKE) run lr=0.01
	$(MAKE) run lr=0.001
	$(MAKE) run lr=0.0001

clean:
	rm -rf weights/*
	rm -rf figures/*

