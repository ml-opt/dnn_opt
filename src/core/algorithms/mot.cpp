#include <algorithm>
#include <stdexcept>
#include <core/algorithms/mot.h>
#include <iostream>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

mot* mot::make(algorithm* base, reader* reader, int k, float beta)
{
  auto* result = new mot(base, reader, k, beta);

  result->init();

  return result;
}

void mot::reset()
{

}

void mot::optimize()
{
  algorithm::optimize();

  _base->optimize();
}

void mot::optimize(int eta, std::function<bool()> on)
{
  bool on_opt = true;
  int span = eta / _k;

  for(int i = 0; i < _k && on_opt; i++)
  {
    set_reader((i + 1) * _beta * _reader->size());

    _base->optimize(span, [&on_opt, &on]()
    {
      on_opt = on();
      return on_opt;
    });
  }
}

void mot::optimize_idev(int count, float dev,  std::function<bool()> on)
{
  bool on_opt = true;

  for(int i = 0; i < _k && on_opt; i++)
  {
    set_reader((i + 1) * _beta * _reader->size());

    _base->optimize_idev(count, dev, [&on_opt, &on]()
    {
      on_opt = on();
      return on_opt;
    });
  }
}

void mot::optimize_dev(float dev,  std::function<bool()> on)
{
  bool on_opt = true;

  for(int i = 0; i < _k && on_opt; i++)
  {
    set_reader((i + 1) * _beta * _reader->size());

    _base->optimize_dev(dev, [&on_opt, &on]()
    {
      on_opt = on();
      return on_opt;
    });
  }
}

void mot::optimize_eval(int count, std::function<bool()> on)
{
  bool on_opt = true;

  for(int i = 0; i < _k && on_opt; i++)
  {
    set_reader((i + 1) * _beta * _reader->size());

    _base->optimize_eval(count / _k,  [&on_opt, &on]()
    {
      on_opt = on();
      return on_opt;
    });
  }
}

solution* mot::get_best()
{
  return _base->get_best();
}

void mot::init()
{
  int n = _network_solutions->size();

  _selector = generators::uniform::make(0.0f, 1.0f);
  _samplers = new proxy_sampler*[n];

  for(int i = 0; i < n; i++)
  {
    _samplers[i] = 0;
  }
}

void mot::set_params(std::vector<float> &params)
{
  _base->set_params(params);
}

void mot::set_reader(int size)
{
  int n = _network_solutions->size();

  _selector->set_max(_reader->size() - size);

  for(int i = 0; i < n; i++)
  {
    int offset = _selector->generate();

    delete _samplers[i];

    _samplers[i] = proxy_sampler::make(_reader, size, offset);
    _network_solutions->get(i)->set_reader(_samplers[i]);
  }
}

mot::mot(algorithm* base, reader* reader, int k, float beta)
: algorithm(base->get_solutions()), _k(k), _beta(beta)
{
  _base = base;
  _network_solutions = get_solutions()->cast_copy<solutions::network>();
  _reader = reader;
}

mot::~mot()
{
  for(int i = 0; i < get_solutions()->size(); i++)
  {
    delete _samplers[i];

    _samplers[i] = 0;
  }

  delete _network_solutions;
  delete _samplers;
  delete _selector;

  _network_solutions = 0;
  _samplers = 0;
  _selector = 0;
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

