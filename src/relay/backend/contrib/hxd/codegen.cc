#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <numeric>
#include <sstream>

#include "../../utils.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

class HXDModuleCodegen : public CSourceModuleCodegenBase,
                         public MemoizedExprTranslator<std::vector<Output>>,
                         public CodegenCBase {
 private:
  std::vector<std::string> GetArgumentNames(const CallNode* call) {
    std::vector<std::string> arg_names;
    for (size_t i = 0; i < call->args.size(); ++i) {
      auto res = VisitExpr(call->args[i]);
      for (const auto& out : res) {
        arg_names.push_back(out.name);
      }
    }
    return arg_names;
  }

  inline size_t GetSize(const Type& type) {
    const auto shape = GetShape(type);
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  }

 public:
  HXDModuleCodegen() : buf_idx_(0) {}
  // CSourceModuleCodegenBase
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    code_stream_ << "#include <cstdint>\n";
    code_stream_ << "#include <cstdlib>\n";
    code_stream_ << "#include <cstring>\n";
    code_stream_ << "#include <vector>\n";
    code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
    code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
    code_stream_ << "#include <dlpack/dlpack.h>\n";
    code_stream_ << "using namespace tvm::runtime;\n";
    code_stream_ << "\n";

    auto func = Downcast<Function>(ref);
    this->ext_func_id_ = GetExtSymbol(func);
    auto out = this->VisitExpr(func->body);
    code_stream_ << JIT(out);
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    return (*pf)(code_stream_.str(), "c", Array<String>{this->ext_func_id_}, Array<String>{});
  }

  // MemoizedExprTranslator
  std::vector<Output> VisitExprDefault_(const Object* op) final {
    LOG(FATAL) << "codegen doesn't support: " << op->GetTypeKey();
    return {};
  }

  std::vector<Output> VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    return {output};
  }

  std::vector<Output> VisitExpr_(const CallNode* call) final {
    LOG(INFO) << "CallNode:" << tvm::PrettyPrint(GetRef<Call>(call));
    auto args = GetArgumentNames(call);
    const auto* op_node = call->op.as<OpNode>();
    const auto op_name = GetRef<Op>(op_node)->name;
    if (op_name == "add") {
      // input
      // auto input_shape = GetShape(call->args[0]->checked_type());
      // output
      auto out_size = GetSize(call->checked_type());

      Output output;
      output.name = "buf_" + std::to_string(buf_idx_++);
      output.size = out_size;
      output.dtype = "float";
      output.need_copy = true;

      ext_func_body_.push_back("for (int i=0; i<" + std::to_string(out_size) + ";i++) {");
      ext_func_body_.push_back("    " + output.name + "[i] = " + args[0] + "[i]+" + args[1] +
                               "[i];");
      ext_func_body_.push_back("}");

      buf_decl_.push_back("float* " + output.name + " = (float*)std::malloc(4 * " +
                          std::to_string(out_size) + ");");
      return {output};
    }
  }

  // CodegenCBase
  std::string JIT(const std::vector<Output>& out) {
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
  }

 private:
  String ext_func_id_;
  Array<Var> ext_func_args_;
  std::vector<std::string> ext_func_body_;
  std::vector<std::string> buf_decl_;
  std::string const_array_name_;
  int buf_idx_;
  std::ostringstream code_stream_;
};

runtime::Module HXDCompiler(const ObjectRef& ref) {
  HXDModuleCodegen hxd;
  return hxd.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.hxd").set_body_typed(HXDCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
