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
class cuckoo : public virtual algorithm
{
public:

  template<class t_solution>
  static cuckoo* make(set<t_solution>* solutions);

  /**
   * @copydoc core::algorithm::reset
   */
  virtual void reset() override;

  /**
   * @copydoc core::algorithm::optimize
   */
  virtual void optimize() override;

  using algorithm::optimize;

  /**
   * @copydoc core::algorithm::get_best
   */
  virtual solution* get_best() override;

  /**
   * @copydoc core::algorithm::set_params
   */
  virtual void set_params(std::vector<float> &params) override;

  float get_scale();

  float get_levy();

  float get_replacement();

  void set_scale(float scale);

  void set_levy(float levy);

  void set_replacement(float replacement);

  virtual ~cuckoo();

protected:

  /**
   * @copydoc core::algorithm::init
   */
  virtual void init() override;

  /**
   * @brief Generates a new candidate solution by performing a levy flight from
   * a given solution.
   *
   * @param index solution from which the levy flight will be performed.
   */
  void generate_new_cuckoo(int cuckoo_idx);

  template<class t_solution>
  cuckoo(set<t_solution>* solutions);

  /** the scale of the optimization problem */
  float _scale;

  /** the levy steep size for the random walk */
  float _levy;

  /** the fraction of worse solutions to be replaced by the algorithm */
  float _replacement;

  /** The solutions optimized by this algorithm */
  set<>* _solutions;

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

template<class t_solution>
cuckoo* cuckoo::make(set<t_solution>* solutions)
{
  auto result = new cuckoo(solutions);

  result->init();

  return result;
}

template<class t_solution>
cuckoo::cuckoo(set<t_solution>* solutions)
: algorithm(solutions)
{
  
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

#endif
