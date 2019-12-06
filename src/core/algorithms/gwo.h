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
 * called Gray Wolf Optimization (GWO). This is a population based algorithm
 * inspired in the hunting behavior of gray wolves.
 *
 * MIRJALILI, Seyedali; MIRJALILI, Seyed Mohammad; LEWIS, Andrew. Grey wolf
 * optimizer. Advances in engineering software, 2014, vol. 69, p. 46-61.
 *
 * @author Randy Alonso Benitez <rbenitez@estudiantes.uci.cu>
 * @version 1.0
 * @date November, 2019
 */
class gwo : public virtual algorithm
{
public:

  template<class t_solution>
  static gwo* make(set<t_solution>* solutions);

  /**
   * @copydoc algorithm::reset()
   *
   * Re-generate the solutions, set hyper-parameters to its defaults values.
   */
  virtual void reset() override;

  virtual void optimize() override;

  using algorithm::optimize;

  /**
   * @brief Update the values ​​of the three best agents.
   */
  virtual void update_elite() override;

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

  solution* _alpha;
  solution* _beta ;
  solution* _delta;
  float* _a ;
  float* _r1;
  float* _r2;
  float* _A1;
  float* _C1;
  float* _r;
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
