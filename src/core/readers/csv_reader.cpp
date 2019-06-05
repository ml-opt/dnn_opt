#include <stdexcept>
#include <limits>
#include <ios>
#include <iostream>
#include <sstream>
#include <core/readers/csv_reader.h>

namespace dnn_opt
{
namespace core
{
namespace readers
{  

csv_reader* csv_reader::make(std::string file_name, int in_dim, int out_dim, char sep, bool header)
{
  return new csv_reader(file_name, in_dim, out_dim, sep, header);
}

float* csv_reader::in_data()
{
  return _in_data;
}

float* csv_reader::out_data()
{
  return _out_data;
}

int csv_reader::get_in_dim() const
{
  return _in_dim;
}

int csv_reader::get_out_dim() const
{
  return _out_dim;
}

int csv_reader::size() const
{
  return _size;
}

csv_reader::~csv_reader()
{
  _file.close();

  delete[] _in_data;
  delete[] _out_data;
}

void csv_reader::load()
{
  if(_header)
  {
    _file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }

  for(int i = 0; i < size(); i++)
  {
    int p = i * _in_dim;
    std::string str;
    std::getline(_file, str);
    std::stringstream stream(str);

    for(int j = 0; j < _in_dim; j++)
    {
      std::getline(stream, str, _sep);
      _in_data[p + j] = stof(str);
    }

    p = i * _out_dim;

    for(int j = 0; j < _out_dim; j++)
    {
      std::getline(stream, str, _sep);
      _out_data[p + j] = stof(str);
    }
  }
}

csv_reader::csv_reader(std::string file_name, int in_dim, int out_dim, char sep, bool header)
: _file(file_name), _in_dim(in_dim), _out_dim(out_dim), _sep(sep), _header(header)
{
  if(!_file)
  {
    throw std::invalid_argument("file_name is invalid");
  }

  _size = get_line_count();
  _in_data  = new float[_size * _in_dim];
  _out_data = new float[_size * _out_dim];

  load();
}

int csv_reader::get_line_count()
{
  int count = 0;

  while (_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'))
  {
    count++;
  }
  
  _file.clear();
  _file.seekg(0, std::ios::beg);

  return count + (_header ? -2 : -1);
}

} // namespace readers
} // namespace core
} // namespace dnn_opt
