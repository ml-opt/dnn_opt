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

#ifndef DNN_OPT_CORE_SOLUTION_H
#define DNN_OPT_CORE_SOLUTION_H

#include <core/base/generator.h>

namespace dnn_opt
{
namespace core
{

/**
 * @brief The solution class is intended as an interface to provide custom
 * solutions to be optimized. The most important feature of a solution is
 * its fitness value that measures its quality.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date June, 2016
 * @version 1.0
 */
class solution
{
public:

  static solution* make(generator* generator, unsigned int size);

  /**
   * @brief Change the value of all the parameters of this solution.
   *
   * @param value the new parameter's value.
   */
  void set(float value);

  /**
   * @brief Change the value of a given parameter.
   *
   * @param index the index of the parameter to be changed.
   *
   * @param value the new parameter's value.
   *
   * @throws std::out_of_bounds if the index is incorrect.
   */
  virtual void set(unsigned int index, float value);

  /**
   * @brief The value of a given parameter.
   *
   * @param index the index of the parameter to be returned.
   *
   * @return the current value of the parameter.
   *
   * @throws assertion if the index is incorrect.
   */
  virtual float get(unsigned int index) const;

  /**
   * @brief The number of parameters of this solution.
   *
   * @return the number of parameters.
   */
  virtual unsigned int size() const;

  /**
   * @brief The parameters of this solution. If the parameters are modified
   * outside this class, ensure to notify it by calling set_modified().
   *
   * @return an array containing the parameters.
   */
  virtual float* get_params() const;

  /**
   * @brief Creates an exact replica of this solution.
   *
   * @return a pointer to a copy of this object.
   */
  virtual solution* clone();

  /**
   * @brief Check if the given object instance is assignable to this solution.
   *
   * @param a solution to check if it is assignable to this solution.
   *
   * @return true if the given solution is the given solution is assignable,
   * false otherwise.
   */
  virtual bool assignable(const solution* s) const;

  virtual void assign(solution* s);

  /**
   * @brief Calculate the fitness of this solution.
   *
   * This function returns a precalculated fitness if there have not been
   * changes in its parameters, otherwise calls calculate_fitness().
   *
   * @return the fitness value of this solution.
   */
  virtual float fitness();

  /**
   * @brief Initialize the parameters of this solution based on the provided
   * @generator.
   */
  virtual void generate();

  /**
   * @brief Change the state of this solution to ensure the calculation of
   * fitness() next time instead of using the stored value from previous
   * calculations. Use this method when the parameters of this solution have
   * changed from outside this class and you want to notify it.
   *
   * @param modified the state of the parameters of this class. Set true when
   * the parameters have changed and the stored fitness value is obsolete,
   * false otherwise.
   */
  virtual void set_modified(bool modified);

  /**
   * @brief Constrain the values of the parameters of this solution. The values 
   * of this solution parameters are setted to [@ref generator::min(), 
   * @ref generator::max()]
   */
  virtual void set_constrains();

  /**
   * @brief Check if this solution has a better fitness value than
   * the specified one considering the specified optimization operation.
   *
   * If maximization, a solution is better than another if its fitness is 
   * higher than the other. If minimization, a solution is better than another
   * if its fitness is lower than the other.
   *
   * @param s the solution to check if is better or not than this one.
   *
   * @param maximization the optimization operation; true for maximization, 
   * false for minimization.
   *
   * @return true if this solution has better fitness value, false otherwise.
   */
  bool is_better_than(solution* s, bool max);

  bool is_modified();

  /**
   * @brief The number of evaluations of the objective function.
   *
   * @return an integer the number of evaluation.
   */
  int get_evaluations();

  /**
   * @brief Allocate memory used by this solution. Derived classes should
   * implement factory pattern and call this method before returning.
   */
  virtual void init();

  /**
   * @brief The @ref generator used to generate random parameters
   * for this solution.
   *
   * @return a pointer to the @ref generator.
   */
  virtual generator* get_generator() const;

  virtual ~solution();

protected:

  /**
   * @brief Calculate the fitness of this solution from its parameters.
   *
   * @return the fitness of this solution.
   */
  virtual float calculate_fitness();

  /**
   * @brief The basic constructor of the solution class.
   */
  solution(generator* generator, unsigned int size);

  /** Specify if the parameters have changed and the fitnes have changed */
  bool _modified;

  /** The latest known fitness produced by calculate_fitness() */
  float _fitness;

  /** The number of calls to the objective function calculate_fitness() */
  int _evaluations;

  /** The generator used to initialize the parameters this solution */
  generator* _generator;

  /** The array containing the parameters of this solution*/
  float* _params;

  /** The amount of parameters of this solution */
  unsigned int _size;
  
};

} // namespace core
} // namespace dnn_opt

#endif
