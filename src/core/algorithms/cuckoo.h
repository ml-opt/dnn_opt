/*
Copyright (c) 2018, Jairo Rojas-Delgado <jrdelgado@uci.cu>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef DNN_OPT_CORE_ALGORITHMS_CUCKOO
#define DNN_OPT_CORE_ALGORITHMS_CUCKOO

#include <vector>
#include <stdexcept>
#include <core/generators/normal.h>
#include <core/generators/uniform.h>
#include <core/base/algorithm.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

/**
 * @brief The cuckoo class implements an optimization metaheuristic algorithm
 * called Cuckoo Search (CS).
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date July, 2016
 * @version 1.0
 */
class cuckoo : public algorithm
{
public:

  template<class t_solution>
  static cuckoo* make(solution_set<t_solution>* solutions)
  {
    auto result = new cuckoo(solutions);

    result->init();

    return result;
  }

  virtual void reset() override
  {

  }

  virtual void optimize() override
  {
    int source_idx = static_cast<int>(_selector->generate());
    auto source = get_solutions()->get(source_idx);

    generate_new_cuckoo(source_idx);

    if(_updated->is_better_than(source, is_maximization()))
    {
      get_solutions()->get(source_idx)->assign(_updated);
    }

    get_solutions()->sort(!is_maximization());

    for(int i = 0; i < _replacement * get_solutions()->size(); i++)
    {
      get_solutions()->get(i)->generate();
    }
  }

  virtual solution* get_best() override
  {
    return get_solutions()->get_best(is_maximization());
  }

  virtual void set_params(std::vector<float> &params) override
  {
    if(params.size() != 3)
    {
      std::invalid_argument("algorithms::cuckoo set_params expect 3 values");
    }

    set_scale(params.at(0));
    set_levy(params.at(1));
    set_replacement(params.at(2));
  }

  void set_scale(float scale)
  {
    _scale = scale;
  }

  void set_levy(float levy)
  {
    _levy = levy;
  }

  void set_replacement(float replacement)
  {
    _replacement = replacement;
  }

  using algorithm::optimize;

  virtual ~cuckoo()
  {
    delete _updated;
    delete _nd_o;
    delete _nd_1;
    delete _selector;

    delete[] _r;
  }

protected:

  virtual void init() override
  {
    /** mantegna algorithm to calculate levy steep size */

    float dividend = tgamma(1 + _levy) * sin(3.14159265f * _levy / 2);
    float divisor = tgamma((1 + _levy) / 2) * _levy * pow(2, (_levy - 1) / 2);
    float omega = pow(dividend / divisor , 1 / _levy);

    _nd_1 = generators::normal::make(0, 1);
    _nd_o = generators::normal::make(0, omega);
    _selector = generators::uniform::make(0, get_solutions()->size());
    _r = new float[get_solutions()->get_dim()];
    _updated = get_solutions()->get(0)->clone();
  }

  /**
   * @brief Generates a new candidate solution by performing a levy flight from
   * a given solution.
   *
   * @param index the index of a solution from who a levy flight will be performed.
   *
   * @return a pointer to the new generated solution.
   */
  void generate_new_cuckoo(int cuckoo_idx)
  {
    auto cuckoo = get_solutions()->get(cuckoo_idx);
    auto best = get_solutions()->get_best(is_maximization());

    float v = _nd_1->generate();
    float u = _nd_o->generate();
    float levy = u / powf(fabs(v), 1 / _levy);

    _nd_1->generate(get_solutions()->get_dim(), _r);

    for(int i = 0; i < get_solutions()->get_dim(); i++)
    {
      float diff = best->get(i) - cuckoo->get(i);

      _updated->set(i, cuckoo->get(i) + _scale * levy * diff * _r[i]);
    }
  }

  template<class t_solution>
  cuckoo(solution_set<t_solution>* solutions)
  : algorithm(solutions)
  {
    _scale = 0.8;
    _levy = 0.8;
    _replacement = 0.3;
  }

  /** the scale of the optimization problem */
  float _scale;

  /** the levy steep size for the random walk */
  float _levy;

  /** the fraction of worse solutions to be replaced by the algorithm */
  float _replacement;

  /** The solutions optimized by this algorithm */
  solution_set<>* _solutions;

  /** Array of random numbers to store random variations in generation steep */
  float* _r;

  /** generated params of new cuckoo */
  solution* _updated;

  /** normal generator of mean = 0 and std. dev. = 1 */
  generators::normal* _nd_1;

  /** normal generator of mean = 0 and std. dev. = omega*/
  generators::normal*  _nd_o;

  /** uniform generator to select random individuals from the population */
  generators::uniform* _selector;

};

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

#endif
