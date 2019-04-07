#include <stdexcept>
#include <algorithm>
#include <cuda/base/proxy_sampler.h>

namespace dnn_opt
{
namespace cuda
{

proxy_sampler* proxy_sampler::make(reader* reader, int limit, int offset)
{
  return new proxy_sampler(reader, limit, offset);
}

proxy_sampler** proxy_sampler::make_fold(reader* reader, int folds, int overlap)
{
  proxy_sampler** proxy_samplers = new proxy_sampler*[folds];
  int sample_size = reader->size() / folds;
  int limit = std::min(reader->size(), sample_size + overlap / 2);

  for(int i = 0; i < folds; i++)
  {
    int offset = std::max(0, i * sample_size - overlap / 2);

    proxy_samplers[i] = new proxy_sampler(reader, limit, offset);
  }

  return proxy_samplers;
}

proxy_sampler** proxy_sampler::make_fold_prop(reader* reader, int folds, float overlap)
{
  int overlap_proportion = reader->size() * overlap;

  if(overlap < 0 || overlap > 1.0f)
  {
    throw std::out_of_range("overlap parameters should be in the range [0, 1]");
  }

  return make_fold(reader, folds, overlap_proportion);
}

proxy_sampler::proxy_sampler(reader* reader, int limit, int offset)
: core::proxy_sampler(reader, limit, offset)
{

}

proxy_sampler::~proxy_sampler()
{

}

}
}
