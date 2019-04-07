#include <stdexcept>
#include <copt/statics/cv.h>

namespace dnn_opt
{
namespace copt
{
namespace statics
{

cv* cv::make(int k, algorithm* base, reader* reader)
{
  cv* result = new cv(k, base, reader);

  result->init();

  return result;
}

algorithm* cv::get_base() const
{
  return _copt_base;
}

shufler* cv::get_shufler() const
{
  return _copt_shufler;
}

reader* cv::get_reader() const
{
  return _copt_reader;
}

reader* cv::get_fold(int idx) const
{
  // TODO: check idx is in range

  return _copt_fold_containers[idx];
}

reader* cv::get_train_data() const
{
  return _copt_train_data;
}

void cv::init()
{
  int k = get_k();

  for(int i = 0; i < k; i++)
  {
    delete _copt_fold_containers[i];
  }

  delete _copt_train_data;
  delete _copt_shufler;
  delete[] _copt_fold_containers;

  _copt_shufler = shufler::make(get_reader());
  _copt_fold_containers = proxy_sampler::make_fold(get_shufler(), k);

  int test_size = get_fold(0)->size();
  int train_size = get_shufler()->size() - test_size;

  _copt_train_data = proxy_sampler::make(get_shufler(), train_size, test_size);

  _copt_shufler->shufle();
}

cv::cv(int k, algorithm* base, reader* reader)
: algorithm(dynamic_cast<set<>*>(base->get_solutions())),
  core::algorithm(base->get_solutions()),
  core::statics::cv(k, base, reader)
{
  _copt_base = base;
  _copt_reader = reader;
  _copt_train_data = 0;
  _copt_shufler = 0;
  _copt_fold_containers = new proxy_sampler*[k];

  for (int i = 0; i < k; i++)
  {
    _copt_fold_containers[i] = 0;
  }
}

cv::~cv()
{
  int k = get_k();

  for(int i = 0; i < k; i++)
  {
    delete _copt_fold_containers[i];
  }

  delete[] _copt_fold_containers;
  delete _copt_shufler;
}

} // namespace statics
} // namespace copt
} // namespace dnn_opt
