# Nouveau Variational Autoencoder (NVAE)
Simple [NVAE](https://arxiv.org/abs/2007.03898) implementation to be used as 
template for training on other datasets. Intended for effective reuse and 
learning purposes. Will be improved on continually.

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

These are some of the most relevant commands for guild:
```
guild tensorboard 1
guild runs
guild retrain model=<model-hash>
```

See the [guild docs](https://my.guild.ai/docs) for more information.
