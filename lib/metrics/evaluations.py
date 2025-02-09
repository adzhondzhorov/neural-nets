def accuracy(actual, predicted):
    return sum([round(pred[0].data) == act[0].data for pred, act in zip(predicted, actual)])/actual.dims()[0]
