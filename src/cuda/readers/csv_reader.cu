#include <cuda/readers/csv_reader.h>

namespace dnn_opt
{
namespace cuda
{
namespace readers
{

csv_reader* csv_reader::make(std::string file_name, int in_dim, int out_dim, char sep, bool header)
{
  return new csv_reader(file_name, in_dim, out_dim, sep, header);
}

float* csv_reader::in_data()
{
  return _dev_in_data;
}

float* csv_reader::out_data()
{
  return _dev_out_data;
}

csv_reader::~csv_reader()
{
  cudaFree(_dev_in_data);
  cudaFree(_dev_out_data);
}

csv_reader::csv_reader(std::string file_name, int in_dim, int out_dim, char sep, bool header)
: core::readers::csv_reader(file_name, in_dim, out_dim, sep, header)
{
  int s_in = size() * _in_dim * sizeof(float);
  int s_out = size() * _out_dim * sizeof(float);

  cudaMalloc(&_dev_in_data, s_in);
  cudaMalloc(&_dev_out_data, s_out);

  cudaMemcpy(_dev_in_data, _in_data, s_in, cudaMemcpyHostToDevice);
  cudaMemcpy(_dev_out_data, _out_data, s_out, cudaMemcpyHostToDevice);
}

} // namespace readers
} // namespace cuda
} // namespace dnn_opt
