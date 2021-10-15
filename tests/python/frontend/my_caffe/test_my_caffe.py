#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-10-15 08:26
import caffe
from caffe.proto import caffe_pb2 as pb
from google.protobuf import text_format
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import utils, graph_executor

init_net = pb.NetParameter()
predict_net = pb.NetParameter()

# load model
with open("test_deploy.prototxt", "r") as f:
    text_format.Merge(f.read(), predict_net)
# load blob
with open("test_solver_iter_1000.caffemodel", "rb") as f:
    init_net.ParseFromString(f.read())

mod, params = relay.frontend.from_caffe(
    init_net, predict_net, {"data": [1, 1]}, {"data": "float32"}
)

target = "llvm"
target_host = "llvm"

dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=0):
    lib = relay.build(mod, target="llvm", params=params)

m = graph_executor.GraphModule(lib["default"](dev))

data = np.random.randn(10, 1, 1)
for d in data:
    m.set_input("data", d)
    m.run()
    print(f"{d} -> {m.get_output(0).numpy()}")
