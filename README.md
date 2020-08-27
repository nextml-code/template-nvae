# Nouveau Variational Autoencoder (NVAE)
Simple NVAE implementation to be used as template for training on other
datasets. Intended for effective reuse and learning purposes. Will be improved
on continually.

## Installation
```
virtualenv venv --python python3.8
source venv/bin/activate
pip install -r requirements.txt
```

## Prepare
```
python prepare.py
python update_splits.py
```

## Training
```
guild run train
```
