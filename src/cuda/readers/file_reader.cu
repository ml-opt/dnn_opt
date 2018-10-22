#include <cuda/readers/file_reader.h>

namespace dnn_opt
{
namespace cuda
{
namespace readers
{

file_reader* file_reader::make(std::string file_name, int batches)
{
  return new file_reader(file_name, batches);
}

float* file_reader::in_data()
{
  return _dev_in_data;
}

float* file_reader::out_data()
{
  return _dev_out_data;
}

file_reader::~file_reader()
{
  cudaFree(_dev_in_data);
  cudaFree(_dev_out_data);
}

file_reader::file_reader(std::string file_name, int batches)
: core::readers::file_reader(file_name, batches)
{
  int s_in = _batch_size * _in_dim * sizeof(float);
  int s_out = _batch_size * _out_dim * sizeof(float);

  cudaMalloc(&_dev_in_data, s_in);
  cudaMalloc(&_dev_out_data, s_out);

  cudaMemcpy(_dev_in_data, _in_data, s_in, cudaMemcpyHostToDevice);
  cudaMemcpy(_dev_out_data, _out_data, s_out, cudaMemcpyHostToDevice);
}

} // namespace readers
} // namespace cuda
} // namespace dnn_opt
