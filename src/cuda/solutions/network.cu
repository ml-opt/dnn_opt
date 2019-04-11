#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <cuda/solutions/network.h>
#include <iostream>
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
  linked* nn = new linked(this);

  for(auto &l : _layers)
  {
    nn->add_layer(l->clone());
  }

  nn->init();

  nn->_fitness = fitness();
  nn->_evaluations = get_evaluations();
  nn->set_modified(false);

  auto params = thrust::device_pointer_cast(get_params());
  auto nn_params = thrust::device_pointer_cast(nn->get_params());

  thrust::copy_n(params, size(), nn_params);

  return nn;
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
  reader* r = get_reader();

  cudaFree(CURRENT_OUT);
  cudaFree(PRIOR_OUT);

  cudaMalloc(&CURRENT_OUT, r->size() * _max_out * sizeof(float));
  cudaMalloc(&PRIOR_OUT, r->size() * _max_out * sizeof(float));

  solution::init();
}

reader* network::get_reader() const
{
  return dynamic_cast<reader*>(_r);
}

void network::set_reader(core::reader* reader)
{
  if(_r->size() < reader->size())
  {
    cudaFree(CURRENT_OUT);
    cudaFree(PRIOR_OUT);

    cudaMalloc(&CURRENT_OUT, reader->size() * _max_out * sizeof(float));
    cudaMalloc(&PRIOR_OUT, reader->size() * _max_out * sizeof(float));
  }

  _r = reader;
  _modified = true;
}

error* network::get_error() const
{
  return dynamic_cast<error*>(_e);
}

float* network::predict(core::reader* validation_set)
{
  int n = validation_set->size() * validation_set->get_out_dim();
  reader* current_reader = get_reader();
  float* result;

  cudaMalloc(&result, n * sizeof(float));

  set_reader(validation_set);
  thrust::copy_n(prop(), n, result);
  set_reader(current_reader);

  return result;
}

network::~network()
{
//  cudaFree(CURRENT_OUT);
//  cudaFree(PRIOR_OUT);
  
//  /* avoid double free from core:network destructor */
//  CURRENT_OUT = 0;
//  PRIOR_OUT = 0;
}

network::network(generator* generator, reader* reader, error* error)
: core::solution(generator, 0) ,
  core::solutions::network(generator, reader, error),
  solution(generator, 0)
{

}

network::network(generator* generator)
: solution(generator, 0),
  core::solution(generator, 0),
  core::solutions::network(generator)
{

}

reader* network::linked::get_reader() const
{
  return _cuda_base->get_reader();
}

void network::linked::set_reader(core::reader* reader)
{
  network::set_reader(reader);
  _cuda_base->set_reader(reader);
}

error* network::linked::get_error() const
{
  return _cuda_base->get_error();
}

network::linked::linked(network* base)
: solution(base->get_generator(), 0),
  network(base->get_generator()),
  core::solution(base->get_generator(), 0),
  core::solutions::network(base->get_generator()),
  core::solutions::network::linked(base)
{
  _cuda_base = base;
}

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt
