#include <cuda/base/sampler.h>
#include <cuda/algorithms/continuation.h>

namespace dnn_opt
{
namespace cuda
{
namespace algorithms
{

continuation* continuation::make(algorithm* base, builder* builder)
{
  auto result = new continuation(base, builder);

  result->init();

  return result;
}

continuation::~continuation()
{

}

continuation::continuation(algorithm* base, builder* builder)
: core::algorithms::continuation(base, builder),
  core::algorithm(base->get_solutions()),
  algorithm(dynamic_cast<solution_set<>*>(base->get_solutions()))
{

}

continuation::random_builder* continuation::random_builder::make(unsigned int k, float beta)
{
  return new random_builder(k, beta);
}

std::vector<core::reader*> continuation::random_builder::build(core::reader* dataset)
{
  std::vector<reader*> sequence;
  std::vector<core::reader*> result;

  sequence.push_back(dynamic_cast<reader*>(dataset));
  result.push_back(dataset);

  for(int i = 1; i < _k; i++)
  {
    sequence.push_back(sampler::make(sequence.back(), _beta));
    result.push_back(sequence.back());
  }

  std::reverse(result.begin(), result.end());

  return result;
}

continuation::random_builder::random_builder(unsigned int k, float beta)
: core::algorithms::continuation::random_builder(k, beta)
{

}

} // namespace algorithms
} // namespace cuda
} // namespace dnn_opt
