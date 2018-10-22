#include <cuda/solutions/network.h>

namespace dnn_opt
{
namespace cuda
{
namespace solutions
{

network* network::make(generator* generator, reader* reader, error* error)
{
  auto* result = new network(generator, reader, error);

  result->init();

  return result;
}

network* network::clone()
{
  return 0;
}

bool network::assignable(const core::solution* s) const
{
  /* Warning: Incomplete method implementation.
   * Check also that contains the same layered structure.
   */

  return size() == s->size();
}

void network::init()
{
  cudaFree(_current_out);
  cudaFree(_prior_out);

  cudaMalloc(&_current_out, _r->size() * _max_out * sizeof(float));
  cudaMalloc(&_prior_out, _r->size() * _max_out * sizeof(float));

  solution::init();
}

reader* network::get_reader()
{
  return dynamic_cast<reader*>(_r);
}

void network::set_reader(core::reader* reader)
{
  _r = reader;
  _modified = true;

  cudaFree(_current_out);
  cudaFree(_prior_out);

  cudaMalloc(&_current_out, _r->size() * _max_out * sizeof(float));
  cudaMalloc(&_prior_out, _r->size() * _max_out * sizeof(float));
}

network::~network()
{
  cudaFree(_current_out);
  cudaFree(_prior_out);
  
  /* avoid double free from core:network destructor */
  _current_out = 0;
  _prior_out = 0;
}

network::network(generator* generator, reader* reader, error* error)
: core::solution(generator, 0) ,
  core::solutions::network(generator, reader, error),
  solution(generator, 0)
{
 
}

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt
