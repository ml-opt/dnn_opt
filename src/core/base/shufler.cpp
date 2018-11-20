#include <algorithm>
#include <core/base/shufler.h>

namespace dnn_opt
{
namespace core
{

shufler* shufler::make(reader* reader)
{
  return new shufler(reader);
}

float* shufler::in_data()
{
  return _in_data;
}

float* shufler::out_data()
{
  return _out_data;
}

int shufler::get_in_dim() const
{
  return _reader->get_in_dim();
}

int shufler::get_out_dim() const
{
  return _reader->get_out_dim();
}

int shufler::size() const
{
  return _reader->size();
}

void shufler::shufle()
{
  int n = size();

  for(int i = n - 1; i > 0; i--)
  {
    _generator->set_max(i);
    swap(i, static_cast<int>(_generator->generate()));
  }
}

void shufler::swap(int i, int j)
{
  /* reserve auxiliary memory to swap */

  float aux[std::max(get_in_dim(), get_out_dim())];

  /* pointers to the i-th and j-th training patterns */

  float* d_i;
  float* d_j;

  /* swap input patterns */

  d_i = _reader->in_data() + (get_in_dim() * i);
  d_j = _reader->in_data() + (get_in_dim() * j);

  std::copy_n(d_i, get_in_dim(), aux);
  std::copy_n(d_j, get_in_dim(), d_i);
  std::copy_n(aux, get_in_dim(), d_j);

  /* swap output patterns */

  d_i = _reader->out_data() + (get_out_dim() * i);
  d_j = _reader->out_data() + (get_out_dim() * j);

  std::copy_n(d_i, get_out_dim(), aux);
  std::copy_n(d_j, get_out_dim(), d_i);
  std::copy_n(aux, get_out_dim(), d_j);
}

shufler::shufler(reader* reader)
{
  _reader = reader;

  _generator = generators::uniform::make(0, 1);
  _in_data = reader->in_data();
  _out_data = reader->out_data();
  _count = 0;
}

shufler::~shufler()
{
  delete _generator;

  _in_data = 0;
  _out_data = 0;
}

} // namespace core
} // namespace dnn_opt
