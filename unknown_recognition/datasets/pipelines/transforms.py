# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from ..builder import PIPELINES

@PIPELINES.register_module()
class LogitMinMaxNormalize(object):
    def __init__(self, method: str="global") -> None:
        valid_method = ["global", "local", "top2", "max", "None"]
        assert method in valid_method, \
            f"method should be in {valid_method}, but get {method}"
        self.method = method

    def __call__(self, results):
        logit = results["logit"]
        if self.method == "global":
            mmax = np.max(logit)
            mmin = np.min(logit)
            logit = (logit - mmin) / (mmax - mmin)
        elif self.method == "local":
            assert len(logit.shape) == 3, \
                f"logit's shape should be [C, H, W], but get {logit.shape}"
            mmax = np.max(logit, axis=0, keepdims=True)
            mmin = np.min(logit, axis=0, keepdims=True)
            logit = (logit - mmin) / (mmax - mmin)
        elif self.method == "top2":
            second_max = np.sort(logit, axis=0)[-2: -1, :, :]
            logit = logit - second_max
        elif self.method == "max":
            mmax = np.max(logit, axis=0, keepdims=True)
            mmax = np.abs(mmax) + 1e-6
            logit /= mmax
        elif self.method == "None":
            pass
        results["logit"] = logit
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(method = {self.method})"
        return repr_str
