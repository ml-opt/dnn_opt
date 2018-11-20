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

#ifndef DNN_OPT_CORE_STATICS_K_FOLD
#define DNN_OPT_CORE_STATICS_K_FOLD

#include <vector>
#include <functional>
#include <core/solutions/network.h>
#include <core/base/algorithm.h>
#include <core/base/proxy_sampler.h>
#include <core/base/shufler.h>

namespace dnn_opt
{
namespace core
{
namespace statics
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
class k_fold : public virtual algorithm
{
public:

  /**
   * @brief Restart the internal state of the algorithm as is no optimization
   * step were made.
   *
   * Some optimization algorithms may store internal state over the course of
   * optimization steeps. This method resets such internal state.
   */
  virtual void reset() override;

  virtual void re_sample();

  virtual void optimize() override;

  /**
   * @brief Perform multiple steps of optimization.
   *
   * @param count number of optimization steps to perform.
   */
  virtual void optimize(int count, std::function<void()> on = [](){}) override;

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
  virtual void optimize_idev(int count, float dev, std::function<void()> on = [](){}) override;

  virtual void optimize_dev(float dev, std::function<void()> on = [](){}) override;

  /**
   * @brief Specify if the optimization algorithm should maximize
   * the objective function of the given solutions.
   *
   * By default, @is_maximization() returns false.
   *
   * @return true if the goal is maximization, false if minimization.
   */
  virtual bool is_maximization() override;

  /**
   * @brief Change the optimization operation performed by this
   * meta-heuristic algorithm.
   *
   * @param maximization true to perform maximization, false to perform
   * minimization.
   */
  virtual void set_maximization(bool maximization) override;

  virtual solution* get_best() override;

  virtual void set_params(std::vector<float> &params) override;

  using algorithm::set_params;

  virtual void set_k(float k);

  virtual float get_k();

  virtual int on_fold(std::function<void(reader*, reader*)> listener);

  virtual void remove_fold_listener(int idx);

  virtual float get_validation_error();

  virtual float get_training_error();

  virtual algorithm* get_base();

  /**
   * The basic destructor of this class.
   */
  virtual ~k_fold();

protected:

  /**
   * @brief Allocate dynamic memory to initialize of this class. Derived classes
   * should implement the factory pattern and call this method before returning.
   */
  virtual void init() override;

  /**
   * @brief The basic contructor for an optimization algorithm.
   */
  k_fold(int k, algorithm* base, reader* reader);

private:

  void set_reader(reader* reader);

  void optimize(std::function<void()> base_optimizer);

  /** Number of folds to generate*/
  int _k;

  algorithm* _base;

  reader* _reader;

  solution_set<solutions::network>* _solutions;

  proxy_sampler** _fold_containers;

  shufler* _shufler;

  std::vector<std::function<void(reader*, reader*)>> _on_fold_listeners;

};

} // namespace statics
} // namespace core
} // namespace dnn_opt

#endif
