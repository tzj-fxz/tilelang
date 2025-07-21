/*!
 * \file tl/op/distributed.h
 * \brief Distributed intrinsics.
 *
 */

#include "op.h"
#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {

/*!
 * \brief tvm intrinsics for getting the PE id
 *
 * int GetPE()
 *
 */
const Op &GetPE();

} // namespace tl
} // namespace tvm
