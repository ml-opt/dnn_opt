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

#ifndef DNN_OPT_CORE_ALGORITHMS_FIREFLY
#define DNN_OPT_CORE_ALGORITHMS_FIREFLY

#include <core/base/solution.h>
#include <core/base/solution_set.h>
#include <core/base/algorithm.h>
#include <core/generators/uniform.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

/**
 * @brief The firefly class implements an optimization metaheuristic algorithm
 * called Firefly Algorithm (FA).
 *
 * This is a population based algorithm inspired in the bio-luminicence of 
 * fireflies.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date July, 2016
 * @version 1.0
 */
class firefly : public virtual algorithm
{
public:

  /**
   * @brief Create a new instance of the firefly class.
   *
   * @param solutions the solution set representing the population to
   * optimize.
   */
  template<class t_solution>
  static firefly* make(const solution_set<t_solution>* solutions);

  /**
   * @copydoc algorithm::reset()
   *
   * Re-generate the solutions. Set hyper-parameters to its defaults values.
   */
  virtual void reset();

  virtual void optimize() override;

  using algorithm::optimize;

  virtual solution* get_best() override;

  virtual void set_params(std::vector<float> &params) override;

  using algorithm::set_params;

  /**
   * @brief Returns the light decay hyper-parameter used by this algorithm.
   *
   * @return the light decay value.
   */
  float get_light_decay() const;

  /**
   * @brief Returns the init bright hyper-parameter used by this algorithm.
   *
   * @return the init bright value.
   */
  float get_init_bright() const;

  /**
   * @brief Returns the random influence hyper-parameter used by this algorithm.
   *
   * @return the rand influence value.
   */
  float get_rand_influence() const;

  /**
   * @brief Returns the random decay hyper-parameter used by this algorithm.
   *
   * @return the random decay value.
   */
  float get_rand_decay() const;

  /**
   * @brief Change the value of the light decay hyper-parameter.
   *
   * @param value the new value of the hyper-parameter.
   */
  void set_light_decay(float value);

  /**
   * @brief Change the value of the init bright hyper-parameter.
   *
   * @param value the new value of the hyper-parameter.
   */
  void set_init_bright(float value);

  /**
   * @brief Change the value of the random influence hyper-parameter.
   *
   * @param value the new value of the hyper-parameter.
   */
  void set_rand_influence(float value);

  /**
   * @brief Change the value of the random decay hyper-parameter.
   *
   * @param value the new value of the hyper-parameter.
   */
  void set_rand_decay(float rand_decay);

  /**
   * The basic destructor of the firefly class.
   */
  virtual ~firefly();

protected:

  virtual void init() override;

  /**
   * @brief Move a given firefly to another.
   *
   * @param s the firefly who is going to move.
   *
   * @param t the destination firefly in what the @ref s firefly
   * will move.
   */
  virtual void move(int s, int t);

  /**
   * @brief The Eucliden distance between two fireflies.
   *
   * @param s the first firefly.
   *
   * @param t the second firefly.
   *
   * @return the euclidean distance between source and target.
   */
  virtual float distance(int s, int t);

  /**
   * @brief Create a new instance of the firefly class.
   *
   * @param solutions the solution set representing the population to
   * optimize.
   */
  template<class t_solution>
  firefly(const solution_set<t_solution>* solutions);

  /** The light decay hyper-parameter */
  float _light_decay;

  /** The init bright hyper-parameter */
  float _init_bright;

  /** The random influence hyper-parameter */
  float _rand_influence;

  /** The random decay hyper-parameter */
  float _rand_decay;

  /** Used to store random values used in the move steep */
  float* _r;

  /** A uniform random generator to be used in the move steep */
  generators::uniform* _generator;

};

/* templated function implementations */

template<class t_solution>
firefly* firefly::make(const solution_set<t_solution>* solutions)
{
  auto* result = new firefly(solutions);

  result->init();

  return result;
}

template<class t_solution>
firefly::firefly(const solution_set<t_solution>* solutions)
: algorithm(solutions)
{
  _light_decay = 1;
  _rand_influence = 0.2;
  _rand_decay = 0.98;
  _init_bright = 1;
  _generator = 0;
  _r = 0;
}

} // namespace algorithms
} // namepsace core
} // namespace dnn_opt

#endif
