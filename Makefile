.PHONY: all data features model demo eval charts test clean serve

all: data features model demo eval charts

data:
	bash scripts/generate_data.sh 700
	python scripts/build_dataset.py

features:
	python scripts/build_features.py

model:
	python scripts/train_model.py

demo:
	python scripts/demo_run.py

eval:
	python eval/run_eval.py
	python eval/uncertainty_sweep.py
	python eval/leakage_ablation.py
	python eval/importance.py

charts:
	python scripts/make_charts.py

test:
	pytest tests/ -q

serve:
	uvicorn api.main:app --reload

clean:
	rm -rf data artifacts/*.pkl artifacts/*.jsonl __pycache__ .pytest_cache
