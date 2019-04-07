#include <stdexcept>
#include <core/statics/cv.h>
#include <iostream>
namespace dnn_opt
{
namespace core
{
namespace statics
{

cv* cv::make(int k, algorithm* base, reader* reader)
{
  cv* result = new cv(k, base, reader);

  result->init();

  return result;
}

void cv::reset()
{
  get_base()->reset();
  get_shufler()->shufle();
}

float cv::get_validation_error()
{

}

float cv::get_training_error()
{

}

void cv::optimize()
{
  algorithm* base = get_base();

  optimize([base]()
  {
    base->optimize();
  });
}

void cv::optimize(int count, std::function<bool()> on)
{
  algorithm* base = get_base();

  optimize([count, &on, base]()
  {
    base->optimize(count, on);
  });
}

void cv::optimize_idev(int count, float dev, std::function<bool()> on)
{
  algorithm* base = get_base();

  optimize([count, dev, &on,base]()
  {
    base->optimize_idev(count, dev, on);
  });
}

void cv::optimize_dev(float dev, std::function<bool()> on)
{
  algorithm* base = get_base();

  optimize([dev, &on, base]()
  {
    base->optimize_dev(dev, on);
  });
}

void cv::optimize_eval(int count, std::function<bool()> on)
{
  algorithm* base = get_base();

  optimize([count, &on, base]()
  {
    base->optimize_eval(count, on);
  });
}

bool cv::is_maximization()
{
  return get_base()->is_maximization();
}

void cv::set_maximization(bool maximization)
{
  get_base()->set_maximization(maximization);
}

solution* cv::get_best()
{
  return get_base()->get_best();
}

void cv::set_params(std::vector<float> &params)
{
  get_base()->set_params(params);
}

void cv::set_k(float k)
{
  _k = k;

  init();
}

float cv::get_k()
{
  return _k;
}

algorithm* cv::get_base() const
{
  return _base;
}

void cv::init()
{
  int k = get_k();

  for(int i = 0; i < _k; i++)
  {
    delete _fold_containers[i];
  }

  delete _train_data;
  delete _shufler;
  delete[] _fold_containers;

  _shufler = shufler::make(_reader);
  _fold_containers = proxy_sampler::make_fold(_shufler, k);

  int test_size = get_fold(0)->size();
  int train_size = get_shufler()->size() - test_size;

  _train_data = proxy_sampler::make(get_shufler(), train_size, test_size);

  _shufler->shufle();
}

int cv::on_fold(std::function<void(int, float, float)> listener)
{
  _on_fold_listeners.push_back(listener);

  return _on_fold_listeners.size() - 1;
}

void cv::remove_fold_listener(int idx)
{
  _on_fold_listeners.erase(_on_fold_listeners.begin() + idx);
}

void cv::set_reader(reader* reader)
{
  auto* solutions = get_solutions();

  for (int i = 0; i < solutions->size(); i++)
  {
    solutions->get(i)->set_reader(reader);
  }
}

shufler* cv::get_shufler() const
{
  return _shufler;
}

reader* cv::get_reader() const
{
  return _reader;
}

reader* cv::get_fold(int idx) const
{
  // TODO: check idx is in range

  return _fold_containers[idx];
}

reader* cv::get_train_data() const
{
  return _train_data;
}

void cv::optimize(std::function<void()> base_optimizer)
{
  int k = get_k();
  reader* test_data = get_fold(0);
  reader* train_data = get_train_data();

  for(int i = 0; i < _k; i++)
  {
    float train_error = 0;
    float test_error = 0;

    set_reader(train_data);

    base_optimizer();

    train_error = get_base()->get_best()->fitness();
    test_error = get_solutions()->get_best(_base->is_maximization())->test(test_data);

    for(auto& listener : _on_fold_listeners)
    {
      listener(i, train_error, test_error);
    }

    if(i + 1 < k)
    {
      test_data->swap(get_fold(i + 1));
    }
  }

  delete train_data;
}

set<solutions::network>* cv::get_solutions()
{
  return _solutions;
}

cv::cv(int k, algorithm* base, reader* reader)
: algorithm(base->get_solutions())
{
  _k = k;
  _base = base;
  _reader = reader;
  _solutions = base->get_solutions()->cast_copy<solutions::network>();
  _shufler = 0;
  _train_data = 0;
  _fold_containers = new proxy_sampler*[k];

  for (int i = 0; i < k; i++)
  {
    _fold_containers[i] = 0;
  }
}

cv::~cv()
{
  int k = get_k();

  for(int i = 0; i < k; i++)
  {
    delete _fold_containers[i];
  }

  delete[] _fold_containers;
  delete _shufler;
  delete _solutions;
}

} // namespace statics
} // namespace core
} // namespace dnn_opt
