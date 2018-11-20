#include <algorithm>
#include <copt/base/shufler.h>

namespace dnn_opt
{
namespace copt
{

shufler* shufler::make(reader* reader)
{
  return new shufler(reader);
}

void shufler::shufle()
{
    int n = _reader->size();

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
: core::shufler(reader)
{

}

} // namespace copt
} // namespace dnn_opt
