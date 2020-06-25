#include <vector>
#include <stdexcept>
#include <core/algorithms/cuckoo.h>
#include <iostream>
namespace dnn_opt
{
namespace core
{
namespace algorithms
{

void cuckoo::reset()
{
  /** mantegna algorithm to calculate levy steep size */

  float aux1 = tgamma(1.0f + m_levy) * sin(3.14159265f * m_levy * 0.5f);
  float aux2 = tgamma((1.0f + m_levy) * 0.5f) * m_levy;
  float aux3 = pow(2.0f , (m_levy - 1.0f) * 0.5f);

  m_nd_o->set_max(pow(aux1 / (aux2 * aux3) , 1.0f / m_levy));
}

void cuckoo::optimize()
{
  int n = get_solutions()->size();

  for(int i = 0; i < n; i++)
  {
    auto source = get_solutions()->get(i);

    generate_new_cuckoo(i);

    if(m_updated->is_better_than(source, is_maximization()))
    {
      int evaluations = source->get_evaluations();

      source->assign(m_updated);

      source->set_evaluations(evaluations + m_updated->get_evaluations());
      m_updated->set_evaluations(0);
    }
  }

  get_solutions()->sort(is_maximization());

  for(int i = n * (1.0f - m_replacement); i < n; i++)
  {
    get_solutions()->get(i)->generate();
  }
}

void cuckoo::generate_new_cuckoo(int cuckoo_idx)
{
  int dim = get_solutions()->get_dim();

  float* cuckoo = get_solutions()->get(cuckoo_idx)->get_params();
  float* params = m_updated->get_params();
  m_nd_1->generate(dim, m_r);

  for(int i = 0; i < dim; i++)
  {
    float v = m_nd_1->generate();
    float u = m_nd_o->generate();
    float levy = u / powf(fabs(v), 1.0f / m_levy);

    params[i] = cuckoo[i] + m_scale * levy;
  }

  m_updated->set_modified(true);
}

solution* cuckoo::get_best()
{
  return get_solutions()->get_best(is_maximization());
}

void cuckoo::set_params(std::vector<float> &params)
{
  if(params.size() != 3)
  {
    std::invalid_argument("algorithms::cuckoo set_params expect 3 values");
  }

  set_scale(params.at(0));
  set_levy(params.at(1));
  set_replacement(params.at(2));
}

float cuckoo::get_scale()
{
  return m_scale;
}

float cuckoo::get_levy()
{
  return m_levy;
}

float cuckoo::get_replacement()
{
  return m_replacement;
}

void cuckoo::set_scale(float scale)
{
  m_scale = scale;
}

void cuckoo::set_levy(float levy)
{
  m_levy = levy;
}

void cuckoo::set_replacement(float replacement)
{
  m_replacement = replacement;
}

void cuckoo::init()
{
  m_scale = 0.8f;
  m_levy = 0.8f;
  m_replacement = 0.3f;

  m_nd_1 = generators::normal::make(0.0f, 1.0f);
  m_nd_o = generators::normal::make(0.0f, 1.0f);
  m_selector = generators::uniform::make(0, get_solutions()->size());
  m_updated = get_solutions()->get(0)->clone();
  m_r = new float[get_solutions()->get_dim()];
}

cuckoo::~cuckoo()
{
  delete m_updated;
  delete m_nd_o;
  delete m_nd_1;
  delete m_selector;

  delete[] m_r;
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt
