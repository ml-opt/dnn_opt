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
#include <core/generators/uniform.h>
#include <core/solutions/network.h>

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

  class wrapper;

  class window_reader;

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

  /** The probablity for a partition to get optimized */
  float _density;

  generators::uniform* _generator;

};

class opwa::wrapper : public virtual solution
{
public:

  static wrapper* make(int index, int count, solution* base);

  virtual void init() override;

  virtual float fitness() override;

  virtual void set(unsigned int index, float value) override;

  virtual float get(unsigned int index) const override;

  virtual unsigned int size() const override;

  virtual float* get_params( ) const override;

  virtual solution* clone() override;

  virtual bool assignable(const solution* s) const override;

  virtual ~wrapper();

protected:

  wrapper(int index, int count, solution* base);

  virtual float calculate_fitness() override;

  int _index;
  int _count;

  int _size;
  int _padding;

  solution*  _base;
};

class opwa::window_reader : public virtual reader
{
public:

  static window_reader* make(int in_dim, int out_dim, int capacity);

  virtual void push(float* in, float* out);

  virtual bool is_full() const;

  virtual float* in_data();

  virtual float* out_data();

  virtual int get_in_dim() const;

  virtual int get_out_dim() const;

  virtual int size() const;

  virtual int capacity() const;

  virtual void init();

  virtual ~window_reader();

protected:

  window_reader(int in_dim, int out_dim, int capacity);

  /** The number of dimensions in the in training signal */
  int _in_dim;

  /** The number of dimensions in the out training signal */
  int _out_dim;

  /** The amount of training patterns currently stored in this reader */
  int _size;

  /** The total amount of training patterns that can be stored in the reader */
  int _capacity;

  /** The loaded in training data from file */
  float*  _in_data;

  /** The loaded out training data from file */
  float*  _out_data;

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
