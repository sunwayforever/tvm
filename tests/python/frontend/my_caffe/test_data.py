#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-10-14 13:11
import caffe
import numpy as np


class TestDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.batch = 10
        pass

    def reshape(self, bottom, top):
        top[0].reshape(10, 1)
        top[1].reshape(10, 1)

    def forward(self, bottom, top):
        x = np.random.rand(self.batch, 1)
        top[0].data[...] = x
        top[1].data[...] = x

    def backward(self, top, propagate_down, bottom):
        pass
