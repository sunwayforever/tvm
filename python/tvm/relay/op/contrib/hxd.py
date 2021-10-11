# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
import tvm.ir
from tvm.relay import transform


def _register_external_op_helper(op_name, supported=True):
    @tvm.ir.register_op_attr(op_name, "target.hxd")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("add")


def partition_for_hxd(mod, params=None, **opts):
    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.AnnotateTarget("hxd"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )
    return seq(mod)
