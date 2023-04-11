# Transformer module for preprocessing

## Based on the understanding of the following classes

- [sklearn API](https://scikit-learn.org/stable/developers/develop.html)
- [sklearn.base.BaseEstimater](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py#L175)
- [sklearn.base.TransformerMixin](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/base.py#L808)
- [set_output for pandas dataframe](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep018/proposal.html)

## TODO

- inplace: inplace parameter for fit_transform is not working as expected. It also shall be applied to all transformers.
