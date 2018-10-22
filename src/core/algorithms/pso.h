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

#ifndef DNN_OPT_CORE_ALGORITHMS_PSO
#define DNN_OPT_CORE_ALGORITHMS_PSO

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
 * @brief The pso class implements an optimization metaheuristic algorithm
 * called Particle Swarm Optimization (PSO). This is a population based
 * algorithm inspired in the movements of swarms.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date July, 2016
 */
class pso : public algorithm
{
public:

  template<class t_solution>
  static pso* make(solution_set<t_solution>* solutions);

  /**
   * @copydoc algorithm::reset()
   *
   * Re-generate the solutions, update the local best solutions and the global
   * best solution accordingly. Set hyper-parameters to its defaults values.
   */
  virtual void reset() override;

  virtual void optimize() override;

  using algorithm::optimize;

  virtual solution* get_best() override;

  virtual void set_params(std::vector<float> &params) override;

  using algorithm::set_params;

  /**
   * @brief Return the local influence hyper-parameter used by this algorithm.
   *
   * @return the local influence value.
   */
  float get_local_param() const;

  /**
   * @brief Return the global influence hyper-parameter used by this algorithm.
   *
   * @return the global influence value.
   */
  float get_global_param() const;

  /**
   * @brief Return the inertia hyper-parameter used by this algorithm.
   *
   * @return the inertia value.
   */
  float get_max_speed_param() const;

  float get_min_speed_param() const;

  /**
   * @brief Change the value of the local influence hyper-parameter.
   *
   * @param value the new value of the hyper-parameter.
   */
  void set_local_param(float value);

  /**
   * @brief Change the value of the global influence hyper-parameter.
   *
   * @param value the new value of the hyper-parameter.
   */
  void set_global_param(float value);

  /**
   * @brief Change the value of the inertia hyper-parameter.
   *
   * @param value the new value of the hyper-parameter.
   */
  void set_max_speed_param(float value);

  void set_min_speed_param(float value);

  /**
   * @brief The basic destructor of this class.
   */
  virtual ~pso() override;

protected:

  virtual void init() override;

  /**
   * @brief Update the speed of a given solution.
   *
   * @param index the index of the solution to be updated.
   */
  void update_speed(int idx);

  /**
   * @brief Update the position of a given solution.
   *
   * @param index the index of the solution to be updated.
   */
  void update_position(int idx);

  /**
   * @brief Update the local best solution of a given solution if the current
   * solution is better than the current best local solution.
   *
   * @param index the index of the solution to be updated.
   */
  void update_local(int idx);

  /**
   * @brief Update the global best if a given solution is better than the
   * current best global solution.
   *
   * @param index the index of the solution to be updated.
   */
  void update_global(int idx );

  /**
   * @brief The basic contructor of a pso class.
   *
   * @param solutions a set of individuals.
   */
  template<class t_solution>
  pso(solution_set<t_solution>* solutions);

  /** Hyper-parameter that measures the local best contribution */
  float _local_param;

  /** Hyper-parameter that measures the global best contribution */
  float _global_param;

  float _min_speed;

  float _max_speed;

  float _current_speed_param;

  float _min_speed_param;

  float _max_speed_param;

  /** The best-so-far solutions */
  solution_set<>* _best_so_far;

  /** The speed of each solution */
  solution_set<>* _speed;

  /** Array to store random values for the @ref update_speed() operation */
  float* _r;

  /** The index of the global best solution */
  int _g_best;

  /** A generator of random values for he @ref update_speed() operation */
  generators::uniform* _generator;

  generators::constant* _speed_generator;

};

/* templated function implementations */

template<class t_solution>
pso* pso::make(solution_set<t_solution> *solutions)
{
  auto* result = new pso(solutions);
  result->init();
  return result;
}

template<class t_solution>
pso::pso(solution_set<t_solution>* solutions)
: algorithm(solutions)
{

}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

#endif
