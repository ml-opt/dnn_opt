#include <stdexcept>
#include <cassert>
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
      part->add(opwa::wrapper::make(i, _count, get_solutions()->get(j)));
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

opwa::wrapper* opwa::wrapper::make(int index, int count, solution* base)
{
  auto* result = new wrapper(index, count, base);

  result->init();

  return result;
}

void opwa::wrapper::init()
{
  if(_index + 1 == _count)
  {
    _size =  _base->size() / _count + _base->size() % _count;
  } else
  {
    _size = _base->size() / _count;
  }

  _padding = _index * (_base->size() / _count);
}

float opwa::wrapper::fitness()
{
  return calculate_fitness();
}

void opwa::wrapper::set(unsigned int index, float value)
{
  assert(index < _size);

  float* params = _base->get_params();
  params[index + _padding] = value;

  set_modified(true);
}

float opwa::wrapper::get(unsigned int index) const
{
  assert(index < _size);

  float* params = _base->get_params();

  return params[index + _padding];
}

unsigned int opwa::wrapper::size() const
{
  return _size;
}

float* opwa::wrapper::get_params( ) const
{
  return _base->get_params() + _padding;
}

solution* opwa::wrapper::clone()
{
  auto result = new wrapper(_index, _count, _base);

  return result;
}

bool opwa::wrapper::assignable(const solution* s) const
{
  return true;

  /* TODO: Check this to include _index and _count comprobation */
}

float opwa::wrapper::calculate_fitness()
{
  return _base->fitness();
}

opwa::wrapper::wrapper(int index, int count, solution* base)
: solution(base->get_generator(), 0)
{
  assert(index >= 0 && index < count);
  assert(count <= base->size());

  _base = base;
  _index = index;
  _count = count;
}

opwa::wrapper::~wrapper()
{

}

opwa::window_reader* opwa::window_reader::make(int in_dim, int out_dim, int capacity)
{
  auto* result = new window_reader(in_dim, out_dim, capacity);
  result->init();
  return result;
}

void opwa::window_reader::push(float* in, float* out)
{
  _size += 1;
  _size %= _capacity;
  std::copy_n(in, _in_dim, _in_data + _size * _in_dim);
  std::copy_n(out, _out_dim, _out_data + _size * _out_dim);
}

bool opwa::window_reader::is_full() const
{
  return _size == _capacity;
}

float* opwa::window_reader::in_data()
{
  return _in_data;
}

float* opwa::window_reader::out_data()
{
  return _out_data;
}

int opwa::window_reader::get_in_dim() const
{
  return _in_dim;
}

int opwa::window_reader::get_out_dim() const
{
  return _out_dim;
}

int opwa::window_reader::size() const
{
  return _size;
}

int opwa::window_reader::capacity() const
{
  return _capacity;
}

void opwa::window_reader::init()
{
  _in_data  = new float[_capacity * _in_dim];
  _out_data = new float[_capacity * _out_dim];
  _size = 0;
}

opwa::window_reader::~window_reader()
{
  delete[] _in_data;
  delete[] _out_data;

  _in_data = 0;
  _out_data = 0;
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

