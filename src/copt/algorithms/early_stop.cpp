#include <algorithm>
#include <stdexcept>
#include <copt/base/proxy_sampler.h>
#include <copt/algorithms/early_stop.h>

namespace dnn_opt
{
namespace copt
{
namespace algorithms
{

early_stop* early_stop::make(algorithm* base, stopper *stopper, reader* reader)
{
  auto* result = new early_stop(base, stopper, reader);

  result->init();

  return result;
}

void early_stop::init()
{
  delete _shufler;

  _shufler = shufler::make(_copt_reader);
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

void early_stop::set_p(float p)
{
  int test_size = _reader->size() * p;
  int train_size = _reader->size() - test_size;

  _p = p;

  delete _train_set;
  delete _test_set;

  _copt_test_set = proxy_sampler::make(_copt_reader, test_size);
  _copt_train_set = proxy_sampler::make(_copt_reader, train_size, test_size);

  _test_set = _copt_test_set;
  _train_set = _copt_train_set;

  set_reader();
}

early_stop::early_stop(algorithm* base, stopper* stopper, reader* reader)
: algorithm(dynamic_cast<set<>*>(base->get_solutions())),
  core::algorithm(base->get_solutions()),
  core::algorithms::early_stop(base, stopper,reader)
{
  _copt_base = base;
  _copt_stopper = stopper;
  _copt_reader = reader;
  _copt_shufler = 0;
  _copt_train_set = 0;
  _copt_test_set = 0;
}

early_stop::~early_stop()
{

}

early_stop::test_increase* early_stop::test_increase::make(int count, bool is_maximization)
{
  return new test_increase(count, is_maximization);
}

bool early_stop::test_increase::stop(float train, float test)
{
  return core::algorithms::early_stop::test_increase::stop(train, test);
}

early_stop::test_increase::test_increase(int count, bool is_maximization)
: core::algorithms::early_stop::test_increase(count, is_maximization)
{

}

} // namespace algorithms
} // namespace copt
} // namespace dnn_opt

