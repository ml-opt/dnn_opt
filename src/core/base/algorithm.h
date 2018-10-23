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

#ifndef DNN_OPT_CORE_ALGORITHM
#define DNN_OPT_CORE_ALGORITHM

#include <vector>
#include <functional>
#include <core/base/solution_set.h>

namespace dnn_opt
{
namespace core
{

/**
 * @brief The algorithm class provides basic functionalities to define and
 * implement custom meta-heuristic algorithms used for optimization. In order
 * to extend the library, new algorithms shuold derive from this class.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date June, 2016
 * @version 1.0
 */
class algorithm
{
public:

  /**
   * @brief Restart the internal state of the algorithm as is no optimization
   * step were made.
   *
   * Some optimization algorithms may store internal state over the course of
   * optimization steeps. This method resets such internal state.
   */
  virtual void reset() = 0;

  /**
   * @brief Perform a single step of optimization.
   */
  virtual void optimize() = 0;

  /**
   * @brief Perform multiple steps of optimization.
   *
   * @param count number of optimization steps to perform.
   */
  virtual void optimize(int count, std::function<void()> on = [](){});

  /**
   * @brief Perform optimization until the best solution does not improve its
   * fitness function more than @ref variance in the last @ref count iterations.
   *
   * @param count number of optimization steps to perform before checking @ref
   * variance improvement.
   *
   * @param variance improvement of the best solution fitness value necesary to
   * continue optimization.
   */
  virtual void optimize_idev(int count, float dev, std::function<void()> on = [](){});

  virtual void optimize_dev(float dev, std::function<void()> on = [](){});

  /**
   * @brief Specify if the optimization algorithm should maximize
   * the objective function of the given solutions.
   *
   * By default, @is_maximization() returns false.
   *
   * @return true if the goal is maximization, false if minimization.
   */
  bool is_maximization();

  /**
   * @brief Change the optimization operation performed by this
   * meta-heuristic algorithm.
   *
   * @param maximization true to perform maximization, false to perform
   * minimization.
   */
  void set_maximization(bool maximization);

  /**
   * @brief Returns the solution_set used by the algorithm to store the
   * population to optimize.
   *
   * @return a pointer to the solution_set.
   */
  solution_set<>* get_solutions() const;

  virtual solution* get_best() = 0;

  virtual void set_params(std::vector<float> &params) = 0;

  virtual void set_params(int n, float* params) final;

  /**
   * The basic destructor of this class.
   */
  virtual ~algorithm();

protected:

  /**
   * @brief Allocate dynamic memory to initialize of this class. Derived classes
   * should implement the factory pattern and call this method before returning.
   */
  virtual void init() = 0;

  /**
   * @brief The basic contructor for an optimization algorithm.
   */
  template<class t_solution>
  algorithm(const solution_set<t_solution>* solutions);

private:

  /** The optimization operation performed by this algorithm */
  bool _maximization;

  solution_set<>* _solutions;

};

template<class t_solution>
algorithm::algorithm(const solution_set<t_solution>* solutions)
{
  _maximization = false;
  _solutions = solutions->cast_copy();
}

} // namespace core
} // namespace dnn_opt

#endif
