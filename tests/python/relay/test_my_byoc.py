#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-10-11 11:00
import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor
from tvm.relay.op.contrib import hxd


def get_demo_mod():
    a = relay.var("a", shape=(1, 10), dtype="float32")
    b = relay.var("b", shape=(1, 10), dtype="float32")
    c = relay.var("c", shape=(1, 10), dtype="float32")
    out = relay.add(relay.add(a, b), c)
    out = relay.multiply(out, c)

    func = relay.Function([a, b, c], out)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    return mod


mod = get_demo_mod()
print("------before------")
print(mod)

mod = hxd.partition_for_hxd(mod)
print("------after------")
print(mod)

with tvm.transform.PassContext(opt_level=2):
    lib = relay.build_module.build(mod, target="llvm", params=None)

# print(lib.graph_json)
# print(lib.lib.imported_modules[1].get_source())

# lib need to be `exported and loaded` due to tvm bug
lib.export_library("/tmp/libhxd.so")
lib = tvm.runtime.load_module("/tmp/libhxd.so")

rt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))

rt_mod.set_input("a", np.ones((1, 10)))
rt_mod.set_input("b", np.ones((1, 10)))
rt_mod.set_input("c", np.ones((1, 10)))
rt_mod.run()
tvm_res = rt_mod.get_output(0).numpy()

print(tvm_res)
