#! /usr/bin/env python
# -*- coding:utf-8 -*-

import mxnet as mx
from mxnet import metric


class Accuracy(metric.Accuracy):
    """Computes accuracy classification score.

        The accuracy score is defined as

        .. math::
            \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i=0}^{n-1}
            \\text{1}(\\hat{y_i} == y_i)

        Parameters
        ----------
        axis : int, default=1
            The axis that represents classes
        name : str
            Name of this metric instance for display.
        output_names : list of str, or None
            Name of predictions that should be used when updating with update_dict.
            By default include all predictions.
        label_names : list of str, or None
            Name of labels that should be used when updating with update_dict.
            By default include all labels.

        Examples
        --------
        >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
        >>> labels   = [mx.nd.array([0, 1, 1])]
        >>> acc = mx.metric.Accuracy()
        >>> acc.update(preds = predicts, labels = labels)
        >>> print acc.get()
        ('accuracy', 0.6666666666666666)
        """

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data with class indices as values, one per sample.

        preds : list of `NDArray`
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        labels, preds = metric.check_label_shapes(labels, preds, True)

        temp = preds[0]
        for i in range(1, len(preds)):
            temp = mx.nd.concat(temp, preds[i], dim=0)
        preds = [temp]

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32')
            label = label.asnumpy().astype('int32')
            # flatten before checking shapes to avoid shape miss match
            label = label.flat
            pred_label = pred_label.flat

            metric.check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label == label).sum()
            self.num_inst += len(pred_label)