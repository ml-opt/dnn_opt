#include <stdexcept>
#include <core/readers/file_reader.h>

namespace dnn_opt
{
namespace core
{
namespace readers
{  

file_reader* file_reader::make(std::string file_name)
{
  return new file_reader(file_name);
}

float* const file_reader::in_data()
{
  return _in_data;
}

float* const file_reader::out_data()
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
  for(int i = 0; i < _size; i++)
  {
    for(int j = 0; j < _in_dim; j++)
    {
      _file >> _in_data[j * _size + i];
    }

    for(int j = 0; j < _out_dim; j++)
    {
      _file >> _out_data[j * _size + i];
    }
  }
}

file_reader::file_reader(std::string file_name)
: _file(file_name)
{
  /* TODO: Check correct _file structure */

  if(!_file)
  {
    throw std::invalid_argument("file_name is invalid");
  }

  _file >> _size;
  _file >> _in_dim;
  _file >> _out_dim;

  _in_data = new float[_size * _in_dim];
  _out_data = new float[_size * _out_dim];

  load();
}

} // namespace readers
} // namespace core
} // namespace dnn_opt
