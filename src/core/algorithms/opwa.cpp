#include <stdexcept>
#include <core/algorithms/opwa.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

void opwa::reset()
{

}

void opwa::optimize()
{
  for(auto& algorithm : _algorithms)
  {
    if(_generator->generate() <= get_density())
    {
      algorithm->optimize();
    }
  }

  for(int i = 0; i < get_solutions()->size(); i++)
  {
    get_solutions()->get(i)->set_modified(true);
  }
}

solution* opwa::get_best()
{
  auto best = get_solutions()->get_best(is_maximization());

  return best;
}

void opwa::init()
{
  _density = 1.0f;
  _generator = generators::uniform::make(0.0f, 1.0f);

  for( int i = 0; i < _count; i++ )
  {
    // TODO: Re-use the container <part> multiple times instead delete it

    auto* part = set<>::make(get_solutions()->size());

    for( int j = 0; j < get_solutions()->size(); j++ )
    {
      part->add(solutions::wrapper::make(i, _count, get_solutions()->get(j)));
    }

    auto wa = std::unique_ptr<algorithm>(_builder(part));
    _algorithms.push_back(std::move(wa));

    delete part;
  }
}

void opwa::set_params(std::vector<float> &params)
{
  if(params.size() != 1)
  {
    std::invalid_argument("algorithms::opwa set_params expect 1 value");
  }

  set_density(params.at(0));
}

float opwa::get_density()
{
  return _density;
}

void opwa::set_density(float density)
{
  if(density < 0 || density > 1)
  {
    throw out_of_range("density is a probability factor in [0,1]");
  }

  _density = density;
}

opwa::~opwa()
{
  for(auto& algorithm : _algorithms)
  {
    algorithm->get_solutions()->clean();
  }
  delete _generator;
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

