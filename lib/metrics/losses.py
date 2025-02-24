def mean_squared_error(actual, predicted):
    return ((actual - predicted) ** 2).col_sum()[0]/ actual.dims()[0]


def negative_log_likelihood(actual, predicted):
    return -(actual * predicted.ln()).row_sum().sum() / actual.dims()[0]


def binary_cross_entropy(actual, predicted):
    return -(actual * predicted.ln() + (1 - actual) * (1 - predicted).ln()).col_sum()[0] / actual.dims()[0]
