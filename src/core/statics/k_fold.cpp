#include <stdexcept>
#include <core/statics/k_fold.h>

namespace dnn_opt
{
namespace core
{
namespace statics
{

void k_fold::reset()
{

}

void k_fold::re_sample()
{
  _shufler->shufle();
}

void k_fold::optimize()
{
  algorithm* base = get_base();

  optimize([base]()
  {
    base->optimize();
  });
}

void k_fold::optimize(int count, std::function<void()> on)
{
  algorithm* base = get_base();

  optimize([count, on, base]()
  {
    base->optimize(count, on);
  });
}

void k_fold::optimize_idev(int count, float dev, std::function<void()> on)
{
  algorithm* base = get_base();

  optimize([count, dev, on,base]()
  {
    base->optimize_idev(count, dev, on);
  });
}

void k_fold::optimize_dev(float dev, std::function<void()> on)
{
  algorithm* base = get_base();

  optimize([dev, on, base]()
  {
    base->optimize_dev(dev, on);
  });
}

bool k_fold::is_maximization()
{
  return get_base()->is_maximization();
}

void k_fold::set_maximization(bool maximization)
{
  get_base()->set_maximization(maximization);
}

solution* k_fold::get_best()
{
  return get_base()->get_best();
}

void k_fold::set_params(std::vector<float> &params)
{
  get_base()->set_params(params);
}

void k_fold::set_k(float k)
{
  _k = k;

  init();
}

float k_fold::get_k()
{
  return _k;
}

algorithm* k_fold::get_base()
{
  return _base;
}

void k_fold::init()
{
  algorithm::init();

  for(int i = 0; i < _k; i++)
  {
    delete _fold_containers[i];
  }

  delete _shufler;
  delete[] _fold_containers;

  _shufler = shufler::make(_reader);
  _fold_containers = proxy_sampler::make_fold(_shufler, _k);

  re_sample();
}

int k_fold::on_fold(std::function<void(reader*, reader*)> listener)
{
  _on_fold_listeners.push_back(listener);

  return _on_fold_listeners.size() - 1;
}

void k_fold::remove_fold_listener(int idx)
{
  _on_fold_listeners.erase(_on_fold_listeners.begin() + idx);
}

void k_fold::set_reader(reader* reader)
{
  for (int i = 0; i < _solutions->size(); ++i)
  {
    _solutions->get(i)->set_reader(reader);
  }
}

void k_fold::optimize(std::function<void()> base_optimizer)
{
  float train_error = 0;
  float val_error = 0;

  int val_size = _fold_containers[0]->size();
  int train_size = _shufler->size() - val_size;

  proxy_sampler* val_data = _fold_containers[0];
  proxy_sampler* train_data = proxy_sampler::make(_shufler, train_size, val_size);

  for(int i = 0; i < _k; i++)
  {
    set_reader(train_data);

    base_optimizer();

    for(auto& listener : _on_fold_listeners)
    {
      listener(train_data, val_data);
    }

    if(i + 1 < _k)
    {
      val_data->swap(_fold_containers[i + 1]);
    }
  }
}

k_fold::k_fold(int k, algorithm* base, reader* reader)
: algorithm(base->get_solutions())
{
  _k = k;
  _base = base;
  _reader = reader;
  _solutions = base->get_solutions()->cast_copy<solutions::network>();
  _shufler = 0;
  _fold_containers = 0;
}

k_fold::~k_fold()
{
  for(int i = 0; i < _k; i++)
  {
    delete _fold_containers[i];
  }

  delete _shufler;
  delete[] _fold_containers;
}

} // namespace statics
} // namespace core
} // namespace dnn_opt
