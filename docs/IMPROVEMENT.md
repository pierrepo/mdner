# IMPROVEMENT

A number of points need to be explored:

- Review the annotations made by hand.
- Create more than 10 replicates to get a general idea of MDNER's performance.
- Establish a sampling strategy when inserting paraphrased texts into the learning set (train + test).
- Optimise the hyperparameters after adding paraphrases.
- Check the rate of unlearned molecules in the validation set, i.e. molecules not present in the learning set but present in the validation set.
- Evaluation of other paraphrase models
- If you want to modify a hyperparameter or parameter in the config file, use the `setup_config()` function in the `scripts/mdner.py` script.