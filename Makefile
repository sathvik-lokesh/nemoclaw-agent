# PYTEST_DISABLE_PLUGIN_AUTOLOAD avoids unrelated system pytest plugins (e.g. a
# sourced ROS 2 install) leaking onto PYTHONPATH and breaking collection.
PY ?= python3
PYTEST = PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest

.PHONY: test demo verify clean help

help:
	@echo "make test    - run the test suite"
	@echo "make demo     - run every scripted scenario through the analyzer"
	@echo "make verify    - plan-verification gate (good plan + broken plan)"
	@echo "make clean    - remove caches and generated traces"

test:
	$(PYTEST) -q

verify:
	$(PY) -m src.verify_plan
	-$(PY) -m src.verify_plan --broken

demo:
	@for s in ok error_propagation dropped_handoff correlated livelock conflicting contract; do \
		$(PY) -m src.run --scenario $$s >/dev/null && $(PY) -m src.analyze results/$$s.jsonl; \
		echo; \
	done

clean:
	rm -rf results/*.jsonl results/*.html .pytest_cache __pycache__ src/**/__pycache__
	find . -name '__pycache__' -type d -exec rm -rf {} +
