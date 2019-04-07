#include <cuda/base/sampler.h>
#include <cuda/base/proxy_sampler.h>
#include <cuda/algorithms/continuation.h>

namespace dnn_opt
{
namespace cuda
{
namespace algorithms
{

continuation* continuation::make(algorithm* base, seq* builder)
{
  auto result = new continuation(base, builder);

  result->init();

  return result;
}

continuation::~continuation()
{

}

continuation::continuation(algorithm* base, seq *builder)
: core::algorithms::continuation(base, builder),
  core::algorithm(base->get_solutions()),
  algorithm(dynamic_cast<set<>*>(base->get_solutions()))
{

}

continuation::descent* continuation::descent::make(reader* dataset, int k, float beta)
{
  auto* result  = new descent(dataset, k, beta);

  result->build();

  return result;
}

void continuation::descent::build()
{
  int k = size();
  int n = _cuda_dataset->size() * get_beta();
  float beta = get_beta();

  _cuda_sequence.push_back(_cuda_dataset);

  for(int i = 1; i < k; i++)
  {
    reader* prior = _cuda_sequence.back();

    _cuda_sequence.push_back(proxy_sampler::make(prior, beta * prior->size()));
  }

  std::reverse(_cuda_sequence.begin(), _cuda_sequence.end());
}

reader* continuation::descent::get(int idx)
{
  // TODO: Check idx range

  return _cuda_sequence.at(idx);
}

continuation::descent::descent(reader* dataset, int k, float beta)
: core::algorithms::continuation::descent(dataset, k, beta)
{
  _cuda_dataset = dataset;
}

continuation::fixed* continuation::fixed::make(reader* dataset, int k, float beta)
{
  auto* result  = new fixed(dataset, k, beta);

  result->build();

  return result;
}

void continuation::fixed::build()
{
  int k = size();
  int n = _cuda_dataset->size() * get_beta();

  _cuda_sequence.push_back(_cuda_dataset);

  for(int i = 1; i < k; i++)
  {
    reader* prior = _cuda_sequence.back();
    reader* sub_set = proxy_sampler::make(prior, prior->size() - n);

    _cuda_sequence.push_back(sub_set);
  }

  std::reverse(_cuda_sequence.begin(), _cuda_sequence.end());
}

reader* continuation::fixed::get(int idx)
{
  // TODO: Check idx range

  return _cuda_sequence.at(idx);
}

continuation::fixed::fixed(reader* dataset, int k, float beta)
  : core::algorithms::continuation::fixed(dataset, k, beta)
{
  _cuda_dataset = dataset;
}

continuation::fixed::~fixed()
{

}


} // namespace algorithms
} // namespace cuda
} // namespace dnn_opt
