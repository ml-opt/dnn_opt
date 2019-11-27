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

#ifndef DNN_OPT_CORE_ALGORITHMS_GWO
#define DNN_OPT_CORE_ALGORITHMS_GWO

#include <core/base/algorithm.h>
#include <core/generators/uniform.h>
#include <core/generators/constant.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

/**
 * @brief The GWO class implements an optimization metaheuristic algorithm
 * called Particle Swarm Optimization (GWO). This is a population based on
 * collective intelligence, algorithm inspired in the hunting behavior of
 * gray wolves.
 * @author Randy Alonso Benitez <rbenitez@estudiantes.uci.cu>
 */
class gwo : public virtual algorithm
{
public:

  template<class t_solution>
  static gwo* make(set<t_solution>* solutions);

  /**
   * @copydoc algorithm::reset()
   *
   * Re-generate the solutions, update the local best solutions and the global
   * best solution accordingly. Set hyper-parameters to its defaults values.
   */
  virtual void reset() override;

  virtual float objfunc() ;

  virtual void optimize() override;

  using algorithm::optimize;

  virtual void update_Elite() override;

  virtual solution* get_best() override;

  virtual void set_params(std::vector<float> &params) override;

  using algorithm::set_params;

  /**
   * @brief The basic destructor of this class.
   */
  virtual ~gwo() override;

protected:

  virtual void init() override;



  /**
   * @brief The basic contructor of a gwo class.
   *
   * @param solutions a set of individuals.
   */
  template<class t_solution>
  gwo(set<t_solution>* solutions);
  solution* _alpha = get_solutions()->get(0);
  solution* _beta = get_solutions()->get(1);
  solution* _delta = get_solutions()->get(2);
};

/* templated function implementations */

template<class t_solution>
gwo* gwo::make(set<t_solution> *solutions)
{
  auto* result = new gwo(solutions);
  result->init();
  return result;
}

template<class t_solution>
gwo::gwo(set<t_solution>* solutions)
: algorithm(solutions)
{

}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

#endif
