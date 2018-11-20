#include <algorithm>
#include <stdexcept>
#include <core/base/reader.h>

namespace dnn_opt
{
namespace core
{

void reader::swap(reader* other)
{
  bool chk_1 = size() != other->size();
  bool chk_2 = get_in_dim() != other->get_in_dim();
  bool chk_3 = get_out_dim() != other->get_out_dim();

  if(chk_1 || chk_2 || chk_3)
  {
    throw std::invalid_argument("the given proxy_reader has different size");
  }

  std::swap_ranges(in_data(), in_data() + get_in_dim() * size(), other->in_data());
  std::swap_ranges(out_data(), out_data() + get_out_dim() * size(), other->out_data());
}

reader::~reader()
{

}

} // namespace core
} // namespace dnn_opt
