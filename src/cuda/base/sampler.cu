#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <core/generators/uniform.h>
#include <cuda/base/sampler.h>

namespace dnn_opt
{
namespace cuda
{

sampler* sampler::make(reader* reader, float sample_proportion)
{
  return new sampler(reader, sample_proportion);
}

sampler** sampler::make(reader* reader, int folds)
{
  sampler** samplers = new sampler*[folds];
  float* random = new float[reader->size()];
  auto* generator = core::generators::uniform::make(0, 1);
  float proportion = 1.0f / folds;

  generator->generate(reader->size(), random);

  for(int i = 0; i < folds; i++)
  {
    bool* mask = new bool[reader->size()];

    for(int j = 0; j < reader->size(); j++)
    {
      mask[j] = random[j] >= i * proportion && random[j] < (i + 1) * proportion;
    }

    samplers[i] = new sampler(reader, mask);
  }

  delete[] random;
  delete generator;

  return samplers;
}

void sampler::sample()
{
  int c = 0; // counter for the currently sampled pattern

  auto in_ptr = thrust::device_pointer_cast(_in_data);
  auto out_ptr = thrust::device_pointer_cast(_out_data);

  auto r_in_ptr = thrust::device_pointer_cast(_reader->in_data());
  auto r_out_ptr = thrust::device_pointer_cast(_reader->out_data());

  for(int i = 0; i < _reader->size(); i++)
  {
    if(_mask[i] == true)
    {
      int s_offset = c * get_in_dim(); // sample offset for input
      int r_offset = i * get_in_dim(); // reader offset for input

      thrust::copy_n(r_in_ptr + r_offset, get_in_dim(), in_ptr + s_offset);

      s_offset = c * get_out_dim(); // sample offset for output
      r_offset = i * get_out_dim(); // reader offset for output

      thrust::copy_n(r_out_ptr + r_offset, get_out_dim(), out_ptr + s_offset);

      c += 1; // update counter considering a new pattern have been sampled
    }
  }
}

sampler::sampler(reader* reader, float sample_proportion)
: core::sampler(reader, sample_proportion)
{

}

sampler::sampler(reader* reader, bool* mask)
: core::sampler(reader, mask)
{

}

sampler::~sampler()
{
  delete[] _mask;
  cudaFree(_in_data);
  cudaFree(_out_data);

  _mask = 0;
  _in_data = 0;
  _out_data = 0;
}

} // namespace cuda
} // namespace dnn_opt

