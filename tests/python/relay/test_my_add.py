#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-08-03 11:11
import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor

shape = (10, 10)
a = relay.var("a", shape=shape, dtype="float32")
b = relay.var("b", shape=shape, dtype="float32")
c = relay.my_add(a, b)
# y = relay.add(y, y)
func = relay.Function([a, b], c)
mod = tvm.IRModule.from_expr(func)
print(mod)

with tvm.transform.PassContext(opt_level=0):
    lib = relay.build_module.build(mod, target="c", params=None)
print(lib.lib.get_source())

with tvm.transform.PassContext(opt_level=0):
    lib = relay.build_module.build(mod, target="llvm", params=None)

rt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
rt_mod.set_input("a", np.ones(shape))
rt_mod.set_input("b", np.ones(shape))
rt_mod.run()
tvm_res = rt_mod.get_output(0).numpy()
print(tvm_res)
