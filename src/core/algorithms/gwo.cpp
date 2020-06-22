#include <stdexcept>
#include <core/algorithms/gwo.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

void gwo::optimize()
{
  algorithm::optimize();

  int dim = get_solutions()->get_dim();
  int size = get_solutions()->size();

  for (int i = 0; i < size; i++)
  {
    update_positions(i);
  }

  update_elite();

  m_a -= 0.01;
}

solution* gwo::get_best()
{
  return m_alpha;
}

void gwo::set_params(std::vector<float> &params)
{

}

void gwo::reset()
{
  m_a = 2.0;
}

void gwo::update_positions(int idx)
{
  int dim = get_solutions()->get_dim();

  float* current = get_solutions()->get(idx)->get_params();
  float* alpha = m_alpha->get_params();
  float* beta = m_beta->get_params();
  float* delta = m_delta->get_params();

  m_generator->generate(3 * dim, m_r1);
  m_generator->generate(3 * dim, m_r2);

  for (int j = 0; j < dim; j++)
  {
    int idx = 3 * j;

    float da = abs(2.0f * m_r2[idx] * alpha[j] - current[j]);
    float db = abs(2.0f * m_r2[idx + 1] * beta[j] - current[j]);
    float dc = abs(2.0f * m_r2[idx + 2] * delta[j] - current[j]);

    float x1 = alpha[j] - (2.0f * m_a * m_r1[idx] - m_a) * da;
    float x2 = beta[j] - (2.0f * m_a * m_r1[idx + 1] - m_a) * db;
    float x3 = delta[j] - (2.0f * m_a * m_r1[idx + 2] - m_a) * dc;

    current[j] = (x1 + x2 + x3) / 3.0f;
  }

  get_solutions()->get(idx)->set_constrains();
}

void gwo::update_elite()
{
  float size = get_solutions()->size();

  for(int i = 0; i < size; i++)
  {
    solution* current = get_solutions()->get(i);

    if (current->is_better_than(m_alpha, is_maximization()))
    {
      m_alpha = current;
    }
    else if (current->is_better_than(m_beta, is_maximization()))
    {
      m_beta = current;
    }
    else if (current->is_better_than(m_delta, is_maximization()))
    {
      m_delta = current;
    }
  }
}

void gwo::init()
{
  int dim = get_solutions()->get_dim();

  m_generator = generators::uniform::make(0.0f, 1.0f);
  m_a = 2.0f;
  m_alpha = get_solutions()->get(0);
  m_beta = get_solutions()->get(1);
  m_delta = get_solutions()->get(2);

  m_r1 = new float[3 * dim];
  m_r2 = new float[3 * dim];
}

gwo::~gwo()
{
  delete m_generator;
  delete[] m_r1;
  delete[] m_r2;
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt
