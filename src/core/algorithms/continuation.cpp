#include <algorithm>
#include <stdexcept>
#include <core/algorithms/continuation.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

continuation* continuation::make(algorithm* base, builder* builder)
{
  auto* result = new continuation(base, builder);

  result->init();

  return result;
}

void continuation::reset()
{

}

void continuation::optimize()
{
  _base->optimize();
}

void continuation::optimize(int eta, std::function<void()> on)
{
  int k = _sequence.size();
  int span = eta / k;

  for(int i = 0; i < k; i++)
  {
    set_reader(i);

    _base->optimize(span, on);
  }
}

void continuation::optimize_idev(int count, float dev,  std::function<void()> on)
{
  int k = _sequence.size();

  for(int i = 0; i < k; i++)
  {
    set_reader(i);

    _base->optimize_idev(count, dev, on);
  }
}

void continuation::optimize_dev(float dev,  std::function<void()> on)
{
  int k = _sequence.size();

  for(int i = 0; i < k; i++)
  {
    set_reader(i);

    _base->optimize_dev(dev, on);
  }
}

void continuation::optimize_eval(int count, std::function<void()> on)
{
  int k = _sequence.size();

  for(int i = 0; i < k; i++)
  {
    set_reader(i);

    _base->optimize_eval(count / k, on);
  }
}

solution* continuation::get_best()
{
  return _base->get_best();
}

void continuation::init()
{
  _sequence = _builder->build(_reader);
  _generator = generators::uniform::make(-1, 1);
  _r = new float[get_solutions()->get_dim()];
}

void continuation::set_reader(int index)
{
  for(int i = 0; i < _network_solutions->size(); i++)
  {
    _network_solutions->get(i)->set_reader(_sequence.at(index));
  }
}

void continuation::set_params(std::vector<float> &params)
{
  if(params.size() != 1)
  {
    std::invalid_argument("algorithms::continuation set_params expect 1 value");
  }

  set_radio_decrease(params.at(0));
}

void continuation::set_radio_decrease(float radio_decrease)
{
  _radio_decrease = radio_decrease;
}

continuation::continuation(algorithm* base, builder* builder)
: algorithm(base->get_solutions())
{
  _base = base;
  _network_solutions = get_solutions()->cast_copy<solutions::network>();
  _reader = _network_solutions->get(0)->get_reader();
  _builder = builder;
}

continuation::~continuation()
{
  delete _network_solutions;
  delete _generator;

  for(int i = 0; i < _sequence.size() - 1; i++)
  {
    delete _sequence.at(i);
  }

  delete[] _r;
}

continuation::random_builder* continuation::random_builder::make(unsigned int k, float beta)
{
  return new random_builder(k, beta);
}

std::vector<reader*> continuation::random_builder::build(reader* dataset)
{
  std::vector<reader*> sequence;

  sequence.push_back(dataset);

  for(int i = 1; i < _k; i++)
  {
    sequence.push_back(sampler::make(sequence.back(), _beta));
  }

  std::reverse(sequence.begin(), sequence.end());

  return sequence;
}

continuation::random_builder::random_builder(unsigned int k, float beta)
{
  _k = k;
  _beta = beta;  
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

