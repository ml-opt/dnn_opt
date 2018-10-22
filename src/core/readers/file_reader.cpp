#include <stdexcept>
#include <core/readers/file_reader.h>

namespace dnn_opt
{
namespace core
{
namespace readers
{  

file_reader* file_reader::make(std::string file_name, int batches)
{
  return new file_reader(file_name, batches);
}

float* file_reader::in_data()
{
  return _in_data;
}

float* file_reader::out_data()
{
  return _out_data;
}

int file_reader::get_in_dim() const
{
  return _in_dim;
}

int file_reader::get_out_dim() const
{
  return _out_dim;
}

int file_reader::size() const
{
  return _size;
}

file_reader::~file_reader()
{
  _file.close();

  delete[] _in_data;
  delete[] _out_data;
}

void file_reader::load()
{
  for(int i = 0; i < _batch_size; i++)
  {
    int p = i * _in_dim;

    for(int j = 0; j < _in_dim; j++)
    {
      _file >> _in_data[p + j];
    }

    p = i * _out_dim;

    for(int j = 0; j < _out_dim; j++)
    {
      _file >> _out_data[p + j];
    }
  }
}

file_reader::file_reader(std::string file_name, int batches)
: _file(file_name)
{
  /* TODO: Check correct _file structure and what happens when _size % batches != 0. */
  if(!_file)
  {
    throw std::invalid_argument("file_name is invalid");
  }

  _file >> _size;
  _file >> _in_dim;
  _file >> _out_dim;

  _batch_size = _size / batches;
  _in_data  = new float[_batch_size * _in_dim];
  _out_data = new float[_batch_size * _out_dim];

  load();
}

} // namespace readers
} // namespace core
} // namespace dnn_opt
