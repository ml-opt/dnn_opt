#include <stdexcept>
#include <core/generators/group.h>

namespace dnn_opt
{
namespace core
{
namespace generators
{

group* group::make(std::initializer_list<std::tuple<int, generator*>> members)
{
  return new group(members);
}

void group::generate(int count, float* params)
{
  int counter = 0;
  int n = _members.size();

  while(counter < count)
  {
    for(int i = 0; i < n && counter < count; i++)
    {
      int capacity = std::get<0>(_members.at(i));
      auto* generator = std::get<1>(_members.at(i));
      int actual = counter + capacity < count ? capacity : count - counter;

      generator->generate(actual, params);
      params += actual;
      counter += actual;
    }
  }
}

float group::generate()
{
  return std::get<1>(_members.at(0))->generate();
}

void group::set_constraints(int count, float* params)
{
  int counter = 0;
  int n = _members.size();

  while(counter < count)
  {
    for(int i = 0; i < n && counter < count; i++)
    {
      int capacity = std::get<0>(_members.at(i));
      auto* generator = std::get<1>(_members.at(i));
      int actual = counter + capacity < count ? capacity : count - counter;

      generator->set_constraints(actual, params);
      params += actual;
      counter += actual;
    }
  }
}

group::group(std::initializer_list<std::tuple<int, generator*>> members)
: generator(0, 0)
{
  _max_gen_count = 0;
  _members.insert(_members.end(), members);

  for(int i = 0; i < _members.size(); i++)
  {
    _max_gen_count += std::get<0>(_members.at(i));
  }
}

group::~group()
{

}

} // namespace generators
} // namespace core
} // namespace dnn_opt
