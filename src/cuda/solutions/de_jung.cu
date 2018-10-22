#include <stdexcept>
#include <cublas.h>
#include <cuda/solutions/de_jung.h>

namespace dnn_opt
{
namespace cuda
{
namespace solutions
{

de_jung* de_jung::make(generator* generator, int size)
{
  auto* result = new de_jung(generator, size);

  result->init();

  return result;
}

de_jung* de_jung::clone()
{
  de_jung* result = make(get_generator(), size());
  int s = size() * sizeof(float);

  cudaMemcpy(result->get_params(), get_params(), s, cudaMemcpyDeviceToDevice);

  return result;
}

bool de_jung::assignable(const core::solution* s) const
{
  return this->size() == s->size();
}

void de_jung::to_core(core::solutions::de_jung* solution) const
{
  int s = size() * sizeof(float);

  if(assignable(solution) == false)
  {
    throw std::invalid_argument("given solution must be assignable");
  }

  cudaMemcpy(solution->get_params(), get_params(), s, cudaMemcpyDeviceToHost);
}

void de_jung::from_core(core::solutions::de_jung* solution)
{
  int s = size() * sizeof(float);

  if(assignable(solution) == false)
  {
    throw std::invalid_argument("given solution must be assignable");
  }

  cudaMemcpy(get_params(), solution->get_params(), s, cudaMemcpyHostToDevice);
  this->set_modified(true);
}

float de_jung::calculate_fitness()
{
  float result = cublasSdot(size(), get_params(), 1, get_params(), 1);

  return result;
}

de_jung::de_jung(generator* generator, int size)
: core::solution(generator, size),
  core::solutions::de_jung(generator, size),
  solution(generator, size)
{

}

de_jung::~de_jung()
{

}

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt
