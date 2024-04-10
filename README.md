# PWCL-Stega

This repo contains the pytorch implementation of PWCL-Stega. 
## Dataset preprocess

```
python data_preprocess.py -d <dataset name> --aug ## for datasets
```

## Training the model
Set the parameters in config.py
```
python train.py
```

## Credits

We took help from the following open source projects and we acknowledge their contributions.
1. Supervised Contrastive Loss <https://github.com/HobbitLong/SupContrast>

