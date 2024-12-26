# Eedi - Mining Misconceptions in Mathematics
This is the takoi part of the 4th place solution for Eedi - Mining Misconceptions in Mathematics.</br>
https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551559

## Data download
Download data to ./data/ from https://www.kaggle.com/c/eedi-mining-misconceptions-in-mathematics/data

Please download the cross-validation splits from the following link to ./output/team. The team is using the same folding method.
https://www.kaggle.com/datasets/takoihiraokazu/kaggle-eedi-fold


## Environment
```
docker-compose up --build
```

## Hardware
GPU: A100 (80GB)

## Misconception Generation
The misconception generation is being executed on a Kaggle notebook. Please run the following notebook:</br>
https://www.kaggle.com/code/takoihiraokazu/exp105-qwen-2-5-32b-awq-make-answer-83bdc7

Please download the output file exp105_train_add_text.csv to the following dir:</br>
./output/kaggle/exp105

## Retrieval
### Model Training for Submission Inference
Please run the following code under ./ </br>
The models trained below were used for test inference.
- exp345.py
### Model Training And Inference for Rerank Data Preparation
Please run the following code under ./ </br>
The models trained below were not used for test inference but were instead used to prepare training data for the later Rerank part.
- exp239.py
- exp240.py
- exp241.py
- exp240_241_ensemble.ipynb
- exp239_240_241_inference.ipynb

## Rerank
### Model Training and Ensemble for Submission Inference
Please run the following code under ./ and execute the Kaggle notebook </br>
The models trained below were used for test inference.
- exp341.py
- exp347.py
- exp348.py
- exp349.py
- https://www.kaggle.com/code/takoihiraokazu/exp341-347-lora-merge
- https://www.kaggle.com/code/takoihiraokazu/exp348-349-lora-merge
