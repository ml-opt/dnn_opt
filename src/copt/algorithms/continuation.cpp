#include <algorithm>
#include <stdexcept>
#include <copt/algorithms/continuation.h>

namespace dnn_opt
{
namespace copt
{
namespace algorithms
{

continuation* continuation::make(algorithm* base, builder* builder)
{
  auto* result = new continuation(base, builder);

  result->init();

  return result;
}

continuation::continuation(algorithm* base, builder* builder)
: algorithm(dynamic_cast<solution_set<>*>(base->get_solutions())),
  core::algorithm(base->get_solutions()),
  core::algorithms::continuation(base, builder)
{

}

} // namespace algorithms
} // namespace copt
} // namespace dnn_opt

