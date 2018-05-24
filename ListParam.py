# -*- coding: utf-8 -*-
from pyspark.accumulators import  AccumulatorParam

class ListParamForFeatureAndLabel(AccumulatorParam):
    def zero(self, value):
        return []
    def addInPlace(self, value1, value2):
        value1.extend(value2)
        return value1

class ListParamForLabeledPoint(AccumulatorParam):
    def zero(self, value):
        return []
    def addInPlace(self, value1, value2):
        value1.extend(value2)
        return value1