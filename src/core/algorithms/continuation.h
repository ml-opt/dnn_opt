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

#ifndef DNN_OPT_CORE_ALGORITHMS_CONTINUATION
#define DNN_OPT_CORE_ALGORITHMS_CONTINUATION

#include <vector>
#include <functional>
#include <core/base/algorithm.h>
#include <core/base/sampler.h>
#include <core/base/reader.h>
#include <core/base/solution_set.h>
#include <core/solutions/network.h>
#include <core/generators/uniform.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

class continuation : public virtual algorithm
{
public:

  /** Forward declaration of the builder of subsets of training patterns */
  class builder;

  /** Forward declaration of a random_builder builder */
  class random_builder;

  static continuation* make(algorithm* base, builder* builder);

  virtual void reset() override;

  virtual void optimize() override;

  virtual void optimize(int eta, std::function<void()> on = [](){}) override;

  virtual void optimize_dev_threshold(float dev) override;

  virtual solution* get_best() override;

  virtual void init() override;

  virtual void set_params(std::vector<float> &params) override;

  using algorithm::set_params;

  virtual void set_radio_decrease(float radio_decrease);

  virtual ~continuation() override;

protected:

  continuation(algorithm* base, builder* builder);

  /** The base algorithm that performs optimization */
  algorithm* _base;

  /** A pointer of @ref get_solutions() that do not degrade to core::solution */
  solution_set<solutions::network>* _network_solutions;

  /** The dataset reader extracted from the first network solution */
  reader* _reader;

  /** The sequence of subsets of training patterns */
  std::vector<reader*> _sequence;

  /** The builder of the sequence of subsets of training patterns */
  builder* _builder;

  generators::uniform* _generator;

  float _radio;

  float _radio_decrease;

private:

  void set_reader(int index);

  float* _r;

};

/**
 * @brief The continuation::builder class is an abstract class to implement
 * custom ways of selecting the representative subset of training patterns.
 *
 * @author Jairo Rojas-Delgado<jrdelgado@uci.cu>
 * @date january, 2018
 * @version 1.0
 */
class continuation::builder
{
public:

  virtual std::vector<reader*> build(reader* dataset) = 0;

};

class continuation::random_builder : public continuation::builder
{
public :

  static random_builder* make(unsigned int k, float beta);

  virtual std::vector<reader*> build(reader* dataset) override;

protected:

  random_builder(unsigned int k, float beta);

  /** Amount of subsets of training patterns */
  unsigned int _k;

  /** Proportion of the i-th subset respect to the (i+1)-th subset*/
  float _beta;

};

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

#endif
