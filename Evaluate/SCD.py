# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import numpy as np

def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    return r

def SCD_function(A, B, F):
    r = abs(corr2(F - B, A)) + abs(corr2(F - A, B))
    return r

