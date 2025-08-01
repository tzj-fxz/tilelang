/*!
 * \file simplify.cc
 * \brief Statement simplifier based on analyzer and remove useless parameters
 * of TL PrimFunc.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "arith/ir_mutator_with_analyzer.h"
#include "tir/analysis/control_flow_graph.h"
#include "tir/analysis/var_use_def_analysis.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace arith;

struct SimplifyConfigNode : public AttrsNodeReflAdapter<SimplifyConfigNode> {
  bool transitively_prove_inequalities;
  bool propagate_knowns_to_prove_conditional;
  bool propagate_knowns_to_simplify_expressions;
  bool convert_boolean_to_and_of_ors;
  bool apply_constraints_to_boolean_branches;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<SimplifyConfigNode>()
        .def_ro("transitively_prove_inequalities",
                &SimplifyConfigNode::transitively_prove_inequalities,
                "If true, simplify conditionals with transitive combinations "
                "of scoped constraints",
                refl::DefaultValue(false))
        .def_ro("propagate_knowns_to_prove_conditional",
                &SimplifyConfigNode::propagate_knowns_to_prove_conditional,
                "If true, known buffer values are propagated and used to "
                "statically prove conditionals",
                refl::DefaultValue(false))
        .def_ro("propagate_knowns_to_simplify_expressions",
                &SimplifyConfigNode::propagate_knowns_to_simplify_expressions,
                "If true, known buffer values are propagated and used to "
                "replace BufferLoad wherever "
                "possible",
                refl::DefaultValue(false))
        .def_ro("convert_boolean_to_and_of_ors",
                &SimplifyConfigNode::convert_boolean_to_and_of_ors,
                "If true, simplify conditionals into an AND of ORs",
                refl::DefaultValue(false))
        .def_ro("apply_constraints_to_boolean_branches",
                &SimplifyConfigNode::apply_constraints_to_boolean_branches,
                "If true, simplify each branch of AND/OR under a constraints "
                "provided by the other "
                "branch",
                refl::DefaultValue(false));
  }
  static constexpr const char *_type_key = "tl.transform.SimplifyConfig";
  TVM_FFI_DECLARE_FINAL_OBJECT_INFO(SimplifyConfigNode, BaseAttrsNode);

  RewriteSimplifier::Extension GetEnabledExtensions() const {
    RewriteSimplifier::Extension flags = RewriteSimplifier::kNone;
    if (transitively_prove_inequalities) {
      flags = RewriteSimplifier::Extension(
          flags | RewriteSimplifier::kTransitivelyProveInequalities);
    }
    if (convert_boolean_to_and_of_ors) {
      flags = RewriteSimplifier::Extension(
          flags | RewriteSimplifier::kConvertBooleanToAndOfOrs);
    }
    if (apply_constraints_to_boolean_branches) {
      flags = RewriteSimplifier::Extension(
          flags | RewriteSimplifier::kApplyConstraintsToBooleanBranches);
    }
    return flags;
  }
};

std::unordered_set<const BufferNode *>
CollectUsedBuffers(const PrimFunc &func) {
  struct Visitor : StmtExprVisitor {
    using StmtExprVisitor::VisitExpr_;
    using StmtExprVisitor::VisitStmt_;

    Visitor(PrimFunc func) : func(func) {}

    void VisitExpr_(const CallNode *op) override {
      for (const auto &arg : op->args) {
        for (const auto &it : func->buffer_map) {
          if (Downcast<PrimExpr>(it.second.get()->data).same_as(arg)) {
            used_in_buffer_def_.insert(it.second.get());
          }
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) override {
      VisitBuffer(op->buffer);
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const BufferStoreNode *op) override {
      VisitBuffer(op->buffer);
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const BlockNode *op) override {
      for (const auto &buffer : op->alloc_buffers) {
        for (const auto &it : func->buffer_map) {
          if (it.second.get()->data.same_as(buffer.get()->data)) {
            used_in_buffer_def_.insert(it.second.get());
          }
        }
      }
      for (const auto &buffer : op->reads) {
        for (const auto &it : func->buffer_map) {
          if (it.second.get()->data.same_as(buffer->buffer.get()->data)) {
            used_in_buffer_def_.insert(it.second.get());
          }
        }
      }
      for (const auto &buffer : op->writes) {
        for (const auto &it : func->buffer_map) {
          if (it.second.get()->data.same_as(buffer->buffer.get()->data)) {
            used_in_buffer_def_.insert(it.second.get());
          }
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitBuffer(const Buffer &buf) {
      // Collect buffers that should remain defined
      VarUseDefAnalyzer usage(Array<Var>{});
      usage(buf->data);
      for (const auto &dim : buf->shape) {
        usage(dim);
      }
      for (const auto &dim : buf->strides) {
        usage(dim);
      }
      usage(buf->elem_offset);

      for (const auto &buffer : usage.buffer_use_count_) {
        if (buffer.second >= 1) {
          used_in_buffer_def_.insert(buffer.first);
        }
      }
      for (const auto &buffer : usage.undefined_buffers_) {
        used_in_buffer_def_.insert(buffer.get());
      }
    }
    PrimFunc func;
    std::unordered_set<const BufferNode *> used_in_buffer_def_;
  };

  Visitor visitor(func);
  visitor(func->body);
  return visitor.used_in_buffer_def_;
}

/* \brief Utility function to collect vars that should be retained. Used in
 * Letstmt Only
 */
std::unordered_set<const VarNode *>
CollectVarsUsedInBufferDefinition(const Stmt &stmt) {
  struct Visitor : StmtExprVisitor {
    using StmtExprVisitor::VisitExpr_;
    using StmtExprVisitor::VisitStmt_;

    void VisitExpr_(const BufferLoadNode *op) override {
      VisitBuffer(op->buffer);
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const BufferStoreNode *op) override {
      VisitBuffer(op->buffer);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitBuffer(const Buffer &buf) {
      // Collect variables that should remain defined
      VarUseDefAnalyzer usage(Array<Var>{});
      usage(buf->data);
      for (const auto &dim : buf->shape) {
        usage(dim);
      }
      for (const auto &dim : buf->strides) {
        usage(dim);
      }
      usage(buf->elem_offset);

      // Track for use in LetStmtNode mutator
      for (const auto &var : usage.undefined_) {
        used_in_buffer_def_.insert(var.get());
      }
    }
    std::unordered_set<const VarNode *> used_in_buffer_def_;
  };

  Visitor visitor;
  visitor(stmt);
  return visitor.used_in_buffer_def_;
}

class SimplifyConfig : public Attrs {
public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SimplifyConfig, Attrs,
                                            SimplifyConfigNode);
};
TVM_FFI_STATIC_INIT_BLOCK({ SimplifyConfigNode::RegisterReflection(); });

TVM_REGISTER_NODE_TYPE(SimplifyConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("tl.Simplify", SimplifyConfig);

class StmtSimplifier : public IRMutatorWithAnalyzer {
public:
  static PrimFunc Apply(PrimFunc func, Analyzer *analyzer,
                        Optional<SimplifyConfig> config_opt = std::nullopt,
                        bool simplify_arguments = false) {
    auto config = config_opt.value_or(AttrsWithDefaultValues<SimplifyConfig>());
    analyzer->rewrite_simplify.SetEnabledExtensions(
        config->GetEnabledExtensions());

    std::optional<ControlFlowGraph> touch_pattern = std::nullopt;
    if (config->propagate_knowns_to_prove_conditional ||
        config->propagate_knowns_to_simplify_expressions) {
      touch_pattern = ControlFlowGraph(func->body);
    }

    std::unordered_set<const VarNode *> used_in_buffer_def =
        CollectVarsUsedInBufferDefinition(func->body);
    StmtSimplifier simplifier(analyzer, config, std::move(touch_pattern),
                              std::move(used_in_buffer_def));
    simplifier.MarkBufferMapShapes(func);
    func.CopyOnWrite()->body = simplifier(func->body);

    // Begin to remove useless var and buffer
    // First get used buffers
    simplifier.used_buffers_ = CollectUsedBuffers(func);

    bool param_updated = false;
    Array<Var> new_params;
    Map<Var, Buffer> new_buffer_map;
    // Check whether each buffer is used
    for (const auto &var : func->params) {
      if (func->buffer_map.find(var) != func->buffer_map.end()) {
        if (simplifier.used_buffers_.find(func->buffer_map[var].get()) !=
            simplifier.used_buffers_.end()) {
          new_params.push_back(var);
          new_buffer_map.Set(var, func->buffer_map[var]);
        } else if (simplifier.used_in_buffer_def_.find(
                       func->buffer_map[var]->data.get()) !=
                   simplifier.used_in_buffer_def_.end()) {
          new_params.push_back(var);
          new_buffer_map.Set(var, func->buffer_map[var]);
        } else {
          param_updated = true;
        }
      }
    }

    if (param_updated) {
      return PrimFunc(new_params, func.CopyOnWrite()->body, func->ret_type,
                      new_buffer_map, func->attrs, func->span);
    } else {
      return func;
    }
  }

private:
  explicit StmtSimplifier(
      Analyzer *analyzer, SimplifyConfig config,
      std::optional<ControlFlowGraph> touch_pattern,
      std::unordered_set<const VarNode *> used_in_buffer_def)
      : IRMutatorWithAnalyzer(analyzer), config_(config),
        touch_pattern_(touch_pattern), used_in_buffer_def_(used_in_buffer_def) {
  }

  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitExpr_;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  PrimExpr VisitExpr(const PrimExpr &expr) final {
    if (config_->propagate_knowns_to_simplify_expressions) {
      return touch_pattern_->SimplifyInContext(expr, current_stmt_.value(),
                                               analyzer_);
    } else {
      return analyzer_->Simplify(expr);
    }
  }

  Stmt Simplify(Stmt stmt) { return operator()(std::move(stmt)); }

  Stmt VisitStmt(const Stmt &stmt) override {
    Optional<Stmt> cache = this->current_stmt_;
    this->current_stmt_ = stmt;
    Stmt output = Parent::VisitStmt(stmt);
    this->current_stmt_ = std::move(cache);
    return output;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    analyzer_->Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    With<ConstraintContext> ctx1(analyzer_, op->loop_var >= op->min);
    With<ConstraintContext> ctx2(analyzer_,
                                 op->loop_var < op->min + op->extent);
    return Parent::VisitStmt_(op);
  }

  bool CanInlineLetStmt(const LetStmtNode *op) {
    if (is_const_number(op->value))
      return true;
    if (op->value.as<VarNode>())
      return true;
    // Won't face the deep expression explosion problem as in Let expression.
    // attempt to inline as much as possible if the value integer type(can be
    // index).
    if (!op->value.dtype().is_int())
      return false;
    return SideEffect(op->value) <= CallEffectKind::kPure;
  }

  Stmt VisitStmt_(const LetStmtNode *op) override {
    PrimExpr value = this->VisitExpr(op->value);
    bool can_inline = CanInlineLetStmt(op);
    if (can_inline) {
      // It is usually fine to discard the let binding because the
      // call to simplify will always inline the var.
      //
      // The exception is when the variable is used in a Buffer's
      // definition, as these are not updated by the simplification.
      // After DeclBuffer is required prior to use of a buffer,
      // simplifying can update the buffer definition as well.  The
      // buffer can only be updated at its point of definition,
      // because the points of use may occur within contexts that
      // allow for additional simplifications (e.g. a buffer of shape
      // [i,j] whose first use occurs within "if i==1" should not have
      // its shape simplified to [1,j]).
      analyzer_->Bind(op->var, value);
    } else if (SideEffect(op->value) <= CallEffectKind::kPure) {
      // Even if we aren't replacing all occurrences, they may be
      // necessary for proving conditional statements.
      non_inlined_bindings_.Set(op->var, value);
    }
    Stmt body = this->VisitStmt(op->body);

    // TODO(Lunderberg): Update the Buffer object as part of
    // DeclBuffer updates, which will first require
    // https://github.com/apache/tvm/pull/14778.
    bool used_in_buffer_def = used_in_buffer_def_.count(op->var.get());

    if (can_inline && !used_in_buffer_def) {
      return body;
    } else if (value.same_as(op->value) && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = this->CopyOnWrite(op);
      n->value = std::move(value);
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const IfThenElseNode *op) override {
    if (Optional<Bool> cond = ProveCondition(op->condition)) {
      if (cond.value()->value) {
        return this->VisitStmt(op->then_case);
      } else if (op->else_case) {
        return this->VisitStmt(op->else_case.value());
      } else {
        return Evaluate(0);
      }
    } else {
      return Parent::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const CallNode *op) override {
    if (op->op.same_as(builtin::if_then_else())) {
      if (Optional<Bool> cond = ProveCondition(op->args[0])) {
        if (cond.value()->value) {
          return this->VisitExpr(op->args[1]);
        } else {
          return this->VisitExpr(op->args[2]);
        }
      }
    }
    return Parent::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const VarNode *op) override {
    used_vars_.insert(op);
    return Parent::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) override {
    auto buffer = op->buffer.get();
    if (used_buffers_.find(buffer) == used_buffers_.end()) {
      used_buffers_.insert(buffer);
    }
    return Parent::VisitExpr_(op);
  }

  // eliminate useless stores
  Stmt VisitStmt_(const BufferStoreNode *op) override {
    BufferStore store = Downcast<BufferStore>(Parent::VisitStmt_(op));
    if (const BufferLoadNode *load = store->value.as<BufferLoadNode>()) {
      if (load->buffer->data.same_as(store->buffer->data) &&
          ArrayDeepEqual(load->indices, store->indices) &&
          tir::ExprDeepEqual()(load->buffer->elem_offset,
                               store->buffer->elem_offset) &&
          ArrayDeepEqual(load->buffer->shape, store->buffer->shape) &&
          ArrayDeepEqual(load->buffer->strides, store->buffer->strides)) {
        return Evaluate(0);
      }
    }
    auto buffer = op->buffer.get();
    if (used_buffers_.find(buffer) == used_buffers_.end()) {
      used_buffers_.insert(buffer);
    }
    return std::move(store);
  }

private:
  bool ArrayDeepEqual(const Array<PrimExpr> &lhs, const Array<PrimExpr> &rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs.size(); i++) {
      if (!tir::ExprDeepEqual()(lhs[i], rhs[i])) {
        return false;
      }
    }
    return true;
  }

  /* \brief Internal utility for checking conditionals
   *
   * Uses more aggressive optimization, such as performing additional
   * inlining and tracking known buffer values.
   */
  Optional<Bool> ProveCondition(PrimExpr condition) const {
    condition = Substitute(condition, non_inlined_bindings_);
    if (config_->propagate_knowns_to_prove_conditional) {
      ICHECK(touch_pattern_.has_value());
      condition = touch_pattern_->SimplifyInContext(
          condition, current_stmt_.value(), analyzer_);
    } else {
      condition = analyzer_->Simplify(condition);
    }
    if (const int64_t *as_int = as_const_int(condition)) {
      return Bool(*as_int);
    } else {
      // May have symbolic, need kSymbolicBound level prover.
      if (analyzer_->CanProve(condition) ||
          analyzer_->CanProve(condition,
                              arith::ProofStrength::kSymbolicBound)) {
        return Bool(true);
      }
      return std::nullopt;
    }
  }

  SimplifyConfig config_;
  std::optional<ControlFlowGraph> touch_pattern_;

  Map<Var, PrimExpr> non_inlined_bindings_;
  Optional<Stmt> current_stmt_{std::nullopt};
  std::unordered_set<const VarNode *> used_in_buffer_def_;
  std::unordered_set<const VarNode *> used_vars_;
  std::unordered_set<const BufferNode *> used_buffers_;
};

using namespace tir::transform;

tvm::transform::Pass Simplify(bool simplify_arguments = true) {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    arith::Analyzer analyzer;
    auto cfg = ctx->GetConfig<SimplifyConfig>("tl.Simplify");
    return StmtSimplifier::Apply(f, &analyzer, cfg, simplify_arguments);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.Simplify", {});
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.Simplify", Simplify);
});

} // namespace tl
} // namespace tvm
