/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file annotate_device_regions.cc
 * \brief Split device function from host.
 */
#include "tir/transforms/ir_utils.h"
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

class DeviceRegionAnnotater : public StmtMutator {
public:
  explicit DeviceRegionAnnotater(Target device_target)
      : device_target_(device_target) {}

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tvm::attr::kTarget) {
      // If a target attribute already exists, use it as-is.
      return GetRef<Stmt>(op);
    } else if (op->attr_key == tir::attr::thread_extent ||
               op->attr_key == tir::attr::pipeline_exec_scope ||
               op->attr_key == tir::attr::device_scope) {
      // These attributes are only allowed in device-side code, so
      // they should be annotated with the function's default target.
      Stmt body = GetRef<Stmt>(op);
      return AttrStmt(device_target_, tvm::attr::kTarget, 0, body);
    } else {
      // All other annotations are ignored
      return StmtMutator::VisitStmt_(op);
    }
  }

private:
  Target device_target_;
};

tvm::transform::Pass AnnotateDeviceRegions() {
  using namespace tir::transform;
  auto pass_func = [](PrimFunc func, IRModule mod,
                      tvm::transform::PassContext ctx) -> PrimFunc {
    auto opt_target = func->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(opt_target) << "AnnotateDeviceRegions: Require the target attribute";
    Target target = opt_target.value();
    Target device_target = target.WithoutHost();

    if (target->GetHost()) {
      if (device_target->kind->name == "c") {
        // Annotate the function with the device target
        auto func_body = func->body;
        func.CopyOnWrite()->body =
            AttrStmt(device_target, tvm::attr::kTarget, 0, func_body);
      }

      DeviceRegionAnnotater mutator(target.WithoutHost());
      func.CopyOnWrite()->body = mutator(func->body);
    }
    return func;
  };

  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateDeviceRegions", {});
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateDeviceRegions",
                        AnnotateDeviceRegions);
});

} // namespace tl
} // namespace tvm
