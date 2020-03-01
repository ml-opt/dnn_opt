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
 * This PSO version version is implemented using an inertia weight that
 * is linearly decreased from @ref get_max_speed_param() to
 * @ref get_min_speed_param(). The also known as constriction PSO is a special
 * case of this version.
 *
 * In future versions of this implementation, it could be a good idea to explore
 * further weight decay strategies:
 *
 * BANSAL, Jagdish Chand, et al. Inertia weight strategies in particle swarm
 * optimization. En 2011 Third world congress on nature and biologically
 * inspired computing. IEEE, 2011. p. 633-640.
 *
 * Chauhan, P., Deep, K. & Pant, M. Novel inertia weight strategies for particle
 * swarm optimization. Memetic Comp. 5, 229–251 (2013). https://doi.org/10.1007/
 * s12293-013-0111-9.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date July, 2016
 */
class pso : public virtual algorithm
{
public:

  template<class t_solution>
  static pso* make(set<t_solution>* solutions);

  /**
   * @copydoc algorithm::reset()
   *
   * Re-generate the solutions, update the local best solutions and the global
   * best solution accordingly. Set hyper-parameters to its defaults values.
   */
  virtual void reset() override;

  /**
   * @copydoc algorithm::optimize()
   *
   * For each particle in the population @ref get_solutions() calculates its
   * corresponding speed and update its position based on this speed. Finally,
   * updates the best-so-far particle and the best-global particle in the
   * the population.
   *
   * The inhertia weight is decreased based on the inertia weight strategy.
   */
  virtual void optimize() override;

  using algorithm::optimize;

  /**
   * @copydoc algorithm::get_best()
   */
  virtual solution* get_best() override;

  /**
   * @copydoc algorithm::set_params()
   */
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
   * @brief Return the max inertia hyper-parameter used by this algorithm.
   *
   * @return the inertia value.
   */
  float get_max_speed_param() const;

  /**
   * @brief Return the min inertia hyper-parameter used by this algorithm.
   *
   * @return the inertia value.
   */
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
   * @brief Change the value of the max inertia hyper-parameter.
   *
   * @param value the new value of the hyper-parameter.
   */
  void set_max_speed_param(float value);

  /**
   * @brief Change the value of the min inertia hyper-parameter.
   *
   * @param value the new value of the hyper-parameter.
   */
  void set_min_speed_param(float value);

  /**
   * @brief The basic destructor of this class.
   */
  virtual ~pso() override;

protected:

  /**
   * @copydoc algorithm::init()
   */
  virtual void init() override;

  /**
   * @brief Update the speed of a given solution.
   *
   * @param index the index of the solution to be updated.
   */
  virtual void update_speed(int idx);

  /**
   * @brief Update the position of a given solution.
   *
   * @param index the index of the solution to be updated.
   */
  virtual void update_position(int idx);

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
  pso(set<t_solution>* solutions);

  /** Hyper-parameter that measures the local best contribution */
  float _local_param;

  /** Hyper-parameter that measures the global best contribution */
  float _global_param;

  /** The current inertia weight */
  float _current_speed_param;

  /** The min inertia weight */
  float _min_speed_param;

  /** The max inertia weight */
  float _max_speed_param;

  /** The best-so-far solutions */
  set<>* _best_so_far;

  /** The speed of each solution */
  set<>* _speed;

  /** Array to store random values for the @ref update_speed() operation */
  float* _r;

  /** The index of the global best solution */
  int _g_best;

  /** A generator of random values for the @ref update_speed() operation */
  generators::uniform* _generator;

  /** A generator of random values for the initial speed */
  generators::constant* _speed_generator;

};

/* templated function implementations */

template<class t_solution>
pso* pso::make(set<t_solution> *solutions)
{
  auto* result = new pso(solutions);
  result->init();
  return result;
}

template<class t_solution>
pso::pso(set<t_solution>* solutions)
: algorithm(solutions)
{
  _best_so_far = 0;
  _speed = 0;
  _generator = 0;
  _speed_generator = 0;
  _r = 0;
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

#endif
