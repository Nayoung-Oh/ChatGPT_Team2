# ChatGPT_Team2 Korean Dialect Understanding of ChatGPT
This is the result of the project "ChatGPT_Team2 Korean Dialect Understanding of ChatGPT"

## main folders
- `aihub dataset main (ver1).py` & `(ver2).py`: use AI Hub dataset to get ChatGPT's response, the result is in `test_ver1_90.json` and `test_ve2_90.json`
- `analysis.py`: to analyze generated results, one of the result is `analysis.csv`
- `generate_dataset.ipynb`: filter the original AI hub dataset, generate useful dataset as `result.json`
- `grammar_experiment.py`: a few experiments, giving the grammar feature as input
- `score.py`: to score the response
    - get the input as `naive_result.csv`, `test.csv`
    - generate `naive_score.csv`, `score.csv`


## JIT_dataset
Related to the JIT dataset

- `analysis.py`: to analyze generated results, one of the result is `analysis.csv`
- `change_dataset_format.py`: change the existing dataset format similar to the previous AI Hub dataset, generate `train_result.json` based on `train.json`
- `archive_dataset_main.py`: to get ChatGPT's response
- `score.py`: to score the response
    - get the input as `archive_result_vanilla.csv`, `archive_result.csv`,`archive_result_with_features.csv`
    - generate `vanilla_score.csv`, `archive_score.csv`, `feature_score.csv`

