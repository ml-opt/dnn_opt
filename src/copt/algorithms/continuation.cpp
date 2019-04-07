#include <algorithm>
#include <stdexcept>
#include <copt/base/proxy_sampler.h>
#include <copt/algorithms/continuation.h>

namespace dnn_opt
{
namespace copt
{
namespace algorithms
{

continuation* continuation::make(algorithm* base, seq* builder)
{
  auto* result = new continuation(base, builder);

  result->init();

  return result;
}

continuation::continuation(algorithm* base, seq* builder)
: algorithm(dynamic_cast<set<>*>(base->get_solutions())),
  core::algorithm(base->get_solutions()),
  core::algorithms::continuation(base, builder)
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
  float beta = get_beta();

  _copt_sequence.push_back(_copt_dataset);

  for(int i = 1; i < k; i++)
  {
    reader* prior = _copt_sequence.back();
    _copt_sequence.push_back(proxy_sampler::make(prior, beta * prior->size()));
  }

  std::reverse(_copt_sequence.begin(), _copt_sequence.end());
}

reader* continuation::descent::get(int idx)
{
  return _copt_sequence.at(idx);
}

continuation::descent::descent(reader* dataset, int k, float beta)
  : core::algorithms::continuation::descent(dataset, k, beta)
{
  _copt_dataset = dataset;
}

continuation::descent::~descent()
{
  for(auto* item : _copt_sequence)
  {
    delete item;
  }
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
  int n = _copt_dataset->size() * get_beta();

  _copt_sequence.push_back(_copt_dataset);

  for(int i = 1; i < k; i++)
  {
    reader* prior = _copt_sequence.back();
    _copt_sequence.push_back(proxy_sampler::make(prior, prior->size() - n));
  }

  std::reverse(_copt_sequence.begin(), _copt_sequence.end());
}

reader* continuation::fixed::get(int idx)
{
  return _copt_sequence.at(idx);
}

continuation::fixed::fixed(reader* dataset, int k, float beta)
  : core::algorithms::continuation::fixed(dataset, k, beta)
{
  _copt_dataset = dataset;
}

continuation::fixed::~fixed()
{
  for(auto* item : _copt_sequence)
  {
    delete item;
  }
}

} // namespace algorithms
} // namespace copt
} // namespace dnn_opt

