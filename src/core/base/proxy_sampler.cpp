#include <stdexcept>
#include <algorithm>
#include <core/base/proxy_sampler.h>

namespace dnn_opt
{
namespace core
{

proxy_sampler* proxy_sampler::make(reader* reader, int limit, int offset)
{
  return new proxy_sampler(reader, limit, offset);
}

proxy_sampler** proxy_sampler::make_fold(reader* reader, int folds, int overlap)
{
  proxy_sampler** proxy_samplers = new proxy_sampler*[folds];
  int sample_size = reader->size() / folds;

  for(int i = 0; i < folds; i++)
  {
    int offset = std::max(0, i * sample_size - overlap / 2);
    int limit = std::min(reader->size(), offset + sample_size + overlap / 2);

    proxy_samplers[i] = new proxy_sampler(reader, limit, offset);
  }

  return proxy_samplers;
}

proxy_sampler** proxy_sampler::make_fold_prop(reader* reader, int folds, float overlap)
{
  int overlap_proportion = reader->size() * overlap;

  if(overlap < 0 || overlap > 1.0f)
  {
    throw std::range_error("overlap parameters should be in the range [0, 1]");
  }

  return make_fold(reader, folds, overlap_proportion);
}

float* proxy_sampler::in_data()
{
  return _reader->in_data() + _offset * get_in_dim();
}

float* proxy_sampler::out_data()
{
  return _reader->out_data() + _offset * get_out_dim();
}

int proxy_sampler::get_in_dim() const
{
  return _reader->get_in_dim();
}

int proxy_sampler::get_out_dim() const
{
  return _reader->get_out_dim();
}

int proxy_sampler::size() const
{
  return _limit;
}

proxy_sampler::proxy_sampler(reader* reader, int limit, int offset)
{
  _reader = reader;
  _limit = limit;
  _offset = offset;
}

proxy_sampler::~proxy_sampler()
{

}

}
}
