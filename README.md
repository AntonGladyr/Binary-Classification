# COMP-551

To run the project the following libraries should be installed:

*
    numpy
*
    argparse
*
    matplotlib
 
To run logistic regression on wine dataset with 7000 iterations and 0.001 learning rate. --s command sets the split for
training dataset.
`python runner.py cancer_dataset --s 0.9 train LR --lr 0.001 --m itr --t 7000`

To run logistic linear discriminant analysis on wine dataset with splitting the dataset by 90% for training set and the
remaining for validation:
`python runner.py wine_dataset --s 0.9 train LDA`

Examples:
`python runner.py cancer_dataset --s 0.9 train LR --lr learning_rate_constant3 --m itr --t 4000`

For validation:
`python runner.py cancer_dataset validate LR_V --lr learning_rate_constant3 --m itr --t 7000`
`python runner.py wine_dataset validate LDA`
`python runner.py cancer_dataset validate LR --lr learning_rate_constant3 --m itr --t 7000`

Run `python runner.py --help` to read more about the parameters.

For `--lr` parameter there are the next options:
learning_rate_constant1     =>       1.0
learning_rate_constant2     =>       0.01
learning_rate_constant3     =>       0.001
learning_rate_constant4     =>       0.0001
learning_rate_constant5     =>       0.00001
learning_rate_constant6     =>       0.000001
learning_rate_constant7     =>       0.0000001
learning_rate_constant8     =>       0.00000001
learning_rate_func1         =>       1/1+k

For visualizing run:
`python plotter.py lr -d cancer_dataset`

