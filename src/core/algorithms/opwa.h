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

#ifndef DNN_OPT_CORE_ALGORITHMS_OPWA
#define DNN_OPT_CORE_ALGORITHMS_OPWA

#include <vector>
#include <memory>
#include <functional>
#include <core/base/algorithm.h>
#include <core/base/set.h>
#include <core/solutions/wrapper.h>
#include <core/generators/uniform.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

class opwa : public virtual algorithm
{
public:

  /**
   * Shorthand for the lambda function which creates wrapped algorithms that
   * operates in partitions.
   */
  typedef std::function<algorithm* (set<>*)> wa;

  template<class t_solution>
  static opwa* make(int count, const set<t_solution>* solutions, wa builder);

  virtual void reset() override;

  virtual void optimize() override;

  using algorithm::optimize;

  virtual solution* get_best() override;

  virtual void init() override;

  virtual void set_params(std::vector<float> &params) override;

  float get_density();

  void set_density(float density);

  virtual ~opwa() override;

protected:

  template<class t_solution>
  opwa(int count, const set<t_solution>* solutions, wa builder);

  template<class t_solution>
  opwa(int count, const set<t_solution>* solutions);

protected:

  /** The amount of partitions to create */
  int _count;

  /** Lambda function that creates wrapped algorithms */
  wa _builder;

  /** A list of wrapped algorithms */
  std::vector<std::unique_ptr<algorithm>> _algorithms;

  /** The probablity for a partition to gets optimized */
  float _density;

  generators::uniform* _generator;

};

template<class t_solution>
opwa* opwa::make(int count, const set<t_solution>* solutions, wa builder)
{
  auto* result = new opwa(count, solutions, builder);

  result->init();

  return result;
}

template<class t_solution>
opwa::opwa(int count, const set<t_solution>* solutions, wa builder)
: algorithm(solutions)
{
  _builder = builder;
  _count = count;
}

template<class t_solution>
opwa::opwa(int count, const set<t_solution>* solutions)
: algorithm(solutions)
{
  _count = count;
}

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

#endif
