#include <math.h>
#include <copt/solutions/ackley.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{

ackley* ackley::make(generator* generator, unsigned int size)
{
  auto* result = new ackley(generator, size);

  result->init();

  return result;
}

float ackley::calculate_fitness()
{
  float summatory_1 = 0;
  float summatory_2 = 0;
  float result      = 0;

  solution::calculate_fitness();
  for(int i = 0; i < size(); i++)
  {
    summatory_1 += pow(get(i), 2);
    summatory_2 += cos(2 * 3.141592653f * get(i));
  }
  result  = -20 * exp(-0.2 * sqrt(summatory_1 / size())) ;
  result += -exp(summatory_2 / size()) + 20 + 2.718281828f;

  return result;
}

ackley::ackley(generator* generator, unsigned int size)
: solution(generator, size),
  core::solution(generator, size),
  core::solutions::ackley(generator, size)
{

}

ackley::~ackley()
{

}

} // namespace solutions
} // namespace copt
} // namespace dnn_opt
