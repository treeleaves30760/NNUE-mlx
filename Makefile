.PHONY: sync test play train selfplay evaluate clean build-accel

sync:
	uv sync --extra dev

test:
	uv run pytest tests/ -v

play:
	uv run python scripts/play.py --game $(GAME) --mode $(MODE)

train:
	uv run python scripts/train.py --game $(GAME) --data $(DATA)

selfplay:
	uv run python scripts/selfplay.py --game $(GAME) --games $(N)

evaluate:
	uv run python scripts/evaluate.py --model1 $(M1) --model2 $(M2) --game $(GAME)

build-accel:
	uv run python setup.py build_ext --inplace

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f src/accel/_nnue_accel*.so
