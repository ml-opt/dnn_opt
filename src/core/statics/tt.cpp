#include <algorithm>
#include <core/statics/tt.h>

namespace dnn_opt
{
namespace core
{
namespace statics
{

void tt::reset()
{

}

float tt::get_error()
{
  int k = get_k();
  float best = 0;

  for(int i = 0; i < k; i++)
  {
    best += _results[i];
  }

  best /= k;

  for(int i = 0; i < _results.size(); i += k)
  {
    float current = 0;

    for(int j = 0; j < k; j++)
    {
      best += _results[j];
    }

    current /= k;

    if(is_maximization())
    {
      best = std::max(best, current);
    } else
    {
      best = std::min(best, current);
    }
  }

  return best;
}

float tt::get_bias()
{
  int k = get_k();
  float tt = 0;
  float best_general = get_error();
  float best_fold[k];

  for(int i = 0; i < k; i++)
  {
    best_fold[k] = _results[i];
  }

  for(int i = 0; i < _results.size(); i += k)
  {
    for(int j = 0; j < k; j++)
    {
      if(is_maximization())
      {
        best_fold[j] = std::max(best_fold[j], _results.at(i + j));
      } else
      {
        best_fold[j] = std::min(best_fold[j], _results.at(i + j));
      }
    }
  }

  for(int i = 0; i < k; i++)
  {
    tt += best_general - best_fold[i];
  }

  tt /= k;

  return tt;
}

void tt::init()
{
  k_fold::init();

  k_fold::on_fold([this](reader* train, reader* val)
  {
    auto* best = dynamic_cast<solutions::network*>(this->get_base()->get_best());

    this->_results.push_back(best->test(val));
  });
}

tt::tt(int k, algorithm* base, reader* reader)
: k_fold(k, base, reader),
  algorithm(base->get_solutions())
{

}

tt::~tt()
{

}

} // namespace statics
} // namespace core
} // namespace dnn_opt
