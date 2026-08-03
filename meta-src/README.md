# Meta-Learning application

The process is composed of:
- feature extraction
- hyperparameter tuning
- meta training and testing

# K-fold 
The K-fold requires a 'split' parameter which can be set to 'val' or 'test'.
Based on this parameter, the iteratior will return the validation or testing part of the fold.


# Experiment

Task have been previously executed.

```shell
# Format meta-features
bash format.sh

# Compute meta-learning experiment
bash run-meta.sh

# Ablation study
bash ablation.sh
```