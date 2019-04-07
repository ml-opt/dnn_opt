#include <algorithm>
#include <stdexcept>
#include <core/base/proxy_sampler.h>
#include <core/algorithms/early_stop.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

early_stop* early_stop::make(algorithm* base, stopper *stopper, reader* reader)
{
  auto* result = new early_stop(base, stopper, reader);

  result->init();

  return result;
}

void early_stop::reset()
{

}

void early_stop::optimize()
{
  _base->optimize();
}

void early_stop::optimize(int eta, std::function<bool()> on)
{
  _base->optimize(eta, [this, &on]()
  {
    bool on_opt = true;
    float train = _network_solutions->get_best(is_maximization())->fitness();
    float test = _network_solutions->get_best(is_maximization())->test(_test_set);

    on_opt &= _stopper->stop(train, test);
    on_opt &= on();

    return on_opt;
  });
}

void early_stop::optimize_idev(int count, float dev,  std::function<bool()> on)
{
  _base->optimize_idev(count, dev, [this, &on]()
  {
    bool on_opt = true;
    float train = _network_solutions->get_best(is_maximization())->fitness();
    float test = _network_solutions->get_best(is_maximization())->test(_test_set);

    on_opt &= _stopper->stop(train, test);
    on_opt &= on();

    return on_opt;
  });
}

void early_stop::optimize_dev(float dev,  std::function<bool()> on)
{
  _base->optimize_dev(dev, [this, &on]()
  {
    bool on_opt = true;
    float train = _network_solutions->get_best(is_maximization())->fitness();
    float test = _network_solutions->get_best(is_maximization())->test(_test_set);

    on_opt &= _stopper->stop(train, test);
    on_opt &= on();

    return on_opt;
  });
}

void early_stop::optimize_eval(int count, std::function<bool()> on)
{
  _base->optimize_eval(count, [this, &on]()
  {
    bool on_opt = true;
    float train = _network_solutions->get_best(is_maximization())->fitness();
    float test = _network_solutions->get_best(is_maximization())->test(_test_set);

    on_opt &= _stopper->stop(train, test);
    on_opt &= on();

    return on_opt;
  });
}

solution* early_stop::get_best()
{
  return _base->get_best();
}

void early_stop::init()
{
  delete _shufler;

  _shufler = shufler::make(_reader);
  _shufler->shufle();

  set_p(0.1f);
}

void early_stop::set_reader()
{
  auto* networks = _network_solutions;
  auto* train_set = _train_set;

  for(int i = 0; i < networks->size(); i++)
  {
    networks->get(i)->set_reader(train_set);
  }
}

void early_stop::set_params(std::vector<float> &params)
{
  _base->set_params(params);
}

float early_stop::get_p()
{
  return _p;
}

void early_stop::set_p(float p)
{
  int test_size = _reader->size() * p;
  int train_size = _reader->size() - test_size;

  _p = p;

  delete _train_set;
  delete _test_set;

  _test_set = proxy_sampler::make(_reader, test_size);
  _train_set = proxy_sampler::make(_reader, train_size, test_size);

  set_reader();
}

early_stop::early_stop(algorithm* base, stopper* stopper, reader* reader)
: algorithm(base->get_solutions())
{
  _base = base;
  _network_solutions = get_solutions()->cast_copy<solutions::network>();
  _stopper = stopper;
  _reader = reader;
  _shufler = 0;
  _train_set = 0;
  _test_set = 0;
}

early_stop::~early_stop()
{
  delete _network_solutions;
  delete _train_set;
  delete _test_set;
  delete _shufler;
}

early_stop::test_increase* early_stop::test_increase::make(int count, bool is_maximization)
{
  return new test_increase(count, is_maximization);
}

bool early_stop::test_increase::stop(float train, float test)
{
  _current += 1;

  if(_current == _count)
  {
    _current = 0;

    if(_prior_test > test && _is_maximization)
    {
      return true;
    }
    if(_prior_test < test && !_is_maximization)
    {
      return true;
    }
  }

  return false;
}

early_stop::test_increase::test_increase(int count, bool is_maximization)
{
  _count = count;
  _current = 0;
  _is_maximization = is_maximization;
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

