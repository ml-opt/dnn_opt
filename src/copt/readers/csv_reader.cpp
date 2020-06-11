#include <stdexcept>
#include <copt/readers/csv_reader.h>

namespace dnn_opt
{
namespace copt
{
namespace readers
{

csv_reader* csv_reader::make(std::string file_name, int in_dim, int out_dim, char sep, bool header)
{
  return new csv_reader(file_name, in_dim, out_dim, sep, header);
}

csv_reader::csv_reader(std::string file_name, int in_dim, int out_dim, char sep, bool header)
: core::readers::csv_reader(file_name, in_dim, out_dim, sep, header)
{

}

csv_reader::~csv_reader()
{

}

} // namespace readers
} // namespace copt
} // namespace dnn_opt
