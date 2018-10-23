#include <algorithm>
#include <stdexcept>
#include <core/base/solution.h>

namespace dnn_opt
{
namespace core
{

solution* solution::make(generator* generator, unsigned int size)
{
  auto* result = new solution(generator, size);

  result->init();

  return result;
}

void solution::set(float value)
{
  for(int i = 0; i < size(); i++)
  {
    set(i, value);
  }
}

void solution::set(unsigned int index, float value)
{
  if(index >= size())
  {
    throw std::out_of_range("solution index out of range");
  }
  _params[index] = value;
  set_modified(true);
}

float solution::get(unsigned int index) const
{
  if(index >= size())
  {
    throw std::out_of_range("solution index out of range");
  }

  return _params[index];
}

unsigned int solution::size() const
{
  return _size;
}

float* solution::get_params() const
{
  return _params;
}

solution* solution::clone()
{
  auto* result = solution::make(get_generator(), size());

  std::copy_n(get_params(), size(), result->get_params());

  result->_fitness = fitness();
  result->_evaluations = get_evaluations();
  result->_modified = false;

  return result;
}

bool solution::assignable(const solution* s) const
{
  return s->size() == size();
}

void solution::assign(solution* s)
{
  if(assignable(s) == false)
  {
    throw new std::invalid_argument("solution is not compatible");
  }

  std::copy_n(s->get_params(), size(), get_params());

  _fitness = s->fitness();
  _evaluations = s->get_evaluations();

  set_modified(false);
}

float solution::fitness()
{
  if (_modified == true)
  {
    _fitness = calculate_fitness();
  }

  set_modified(false);

  return _fitness;
}

void solution::generate()
{
  set_modified(true);
  get_generator()->generate(size(), get_params());
}

void solution::set_modified(bool modified)
{
  _modified = modified;
}

void solution::set_constrains()
{
  float min = get_generator()->get_min();
  float max = get_generator()->get_max();
  float* params = get_params();
  int dim = size();

  for(int i = 0; i < dim; i++)
  {
    params[i] = std::min(max, std::max(min, params[i]));
  }

  set_modified(true);
}

bool solution::is_better_than(solution* s, bool max)
{
  return !(max ^ fitness() > s->fitness()) && fitness() != s->fitness();
}

bool solution::is_modified()
{
  return _modified;
}

int solution::get_evaluations()
{
  return _evaluations;
}

void solution::init()
{
  delete[] _params;

  _params = new float[_size];
  _evaluations = 0;

  set_modified(true);
}

generator* solution::get_generator() const
{
  return _generator;
}

float solution::calculate_fitness()
{
  _evaluations += 1;

  return 0;
}

solution::solution(generator* generator, unsigned int size)
{
  _generator = generator;
  _size = size;
  _evaluations = 0;
  _params = 0;

  set_modified(true);
}

solution::~solution()
{
  delete[] _params;
}

} // namespace core
} // namespace dnn_opt
