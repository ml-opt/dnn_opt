#include <algorithm>
#include <stdexcept>
#include <core/algorithms/continuation.h>
#include <iostream>
namespace dnn_opt
{
namespace core
{
namespace algorithms
{

continuation* continuation::make(algorithm* base, seq* builder)
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

void continuation::optimize(int eta, std::function<bool()> on)
{
  bool on_opt = true;
  int k = _builder->size();
  int span = eta / k;

  for(int i = 0; i < k && on_opt; i++)
  {
    set_reader(i);

    _base->optimize(span, [&on_opt, &on]()
    {
      on_opt = on();
      return on_opt;
    });
  }
}

void continuation::optimize_idev(int count, float dev,  std::function<bool()> on)
{
  bool on_opt = true;
  int k = _builder->size();

  for(int i = 0; i < k && on_opt; i++)
  {
    set_reader(i);

    _base->optimize_idev(count, dev, [&on_opt, &on]()
    {
      on_opt = on();
      return on_opt;
    });
  }
}

void continuation::optimize_dev(float dev,  std::function<bool()> on)
{
  bool on_opt = true;
  int k = _builder->size();

  for(int i = 0; i < k && on_opt; i++)
  {
    set_reader(i);

    _base->optimize_dev(dev, [&on_opt, &on]()
    {
      on_opt = on();
      return on_opt;
    });
  }
}

void continuation::optimize_eval(int count, std::function<bool()> on)
{
  bool on_opt = true;
  int k = _builder->size();

  for(int i = 0; i < k && on_opt; i++)
  {
    set_reader(i);

    _base->optimize_eval(count / k,  [&on_opt, &on]()
    {
      on_opt = on();
      return on_opt;
    });

  }
}

solution* continuation::get_best()
{
  return _base->get_best();
}

void continuation::init()
{

}

void continuation::set_reader(int index)
{
  for(int i = 0; i < _network_solutions->size(); i++)
  {
    _network_solutions->get(i)->set_reader(_builder->get(index));
  }
}

void continuation::set_params(std::vector<float> &params)
{
  _base->set_params(params);
}

continuation::continuation(algorithm* base, seq* builder)
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
  delete _builder;
}

continuation::descent* continuation::descent::make(reader* dataset, int k, float beta)
{
  auto* result  = new descent(dataset, k, beta);

  result->build();

  return result;
}

void continuation::descent::build()
{
  int k = size();

  _sequence.push_back(_dataset);

  for(int i = 1; i < k; i++)
  {
    reader* prior = _sequence.back();
    _sequence.push_back(proxy_sampler::make(prior, _beta * prior->size()));
  }

  std::reverse(_sequence.begin(), _sequence.end());
}

reader* continuation::descent::get(int idx)
{
  // TODO: Check idx range

  return _sequence.at(idx);
}

float continuation::descent::get_beta()
{
  return _beta;
}

int continuation::descent::size()
{
  return _k;
}

continuation::descent::descent(reader* dataset, int k, float beta)
{
  _dataset = dataset;
  _k = k;
  _beta = beta;
}

continuation::descent::~descent()
{
  for(auto* item : _sequence)
  {
    delete item;
  }
}

continuation::fixed* continuation::fixed::make(reader* dataset, int k, float beta)
{
  auto* result  = new fixed(dataset, k, beta);

  result->build();

  return result;
}

void continuation::fixed::build()
{
  int k = size();
  int n = _dataset->size() * get_beta();

  _sequence.push_back(_dataset);

  for(int i = 1; i < k; i++)
  {
    reader* prior = _sequence.back();
    _sequence.push_back(proxy_sampler::make(prior, prior->size() - n));
  }

  std::reverse(_sequence.begin(), _sequence.end());
}

reader* continuation::fixed::get(int idx)
{
  // TODO: Check idx range

  return _sequence.at(idx);
}

float continuation::fixed::get_beta()
{
  return _beta;
}

int continuation::fixed::size()
{
  return _k;
}

continuation::fixed::fixed(reader* dataset, int k, float beta)
{
  _dataset = dataset;
  _k = k;
  _beta = beta;
}

continuation::fixed::~fixed()
{
  for(auto* item : _sequence)
  {
    delete item;
  }
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

