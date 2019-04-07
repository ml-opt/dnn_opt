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

#ifndef DNN_OPT_CORE_SET
#define DNN_OPT_CORE_SET

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <core/base/solution.h>

using namespace std;

namespace dnn_opt
{
namespace core
{

/**
 * @brief The set class is intended to manage a set of optimization
 * solutions for a determined optimization problem.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date June, 2016
 * @version 1.0
 */
template<class t_solution = solution>
class set
{

static_assert(true, "t_solution must derive from core::solution");

public:

  static set<t_solution>* make(unsigned int size = 10);

  /**
   * @brief The number of solutions of this container.
   *
   * @return the number of solutions of this container.
   */
  int size() const;

  /**
   * @brief The number of dimensions of the solutions in this container.
   *
   * @return the number of dimensions of the solutions in this container.
   */
  int get_dim() const;

  /**
   * @brief Return a pointer to a specified solution.
   *
   * @param index the index of a solution in this container.
   *
   * @return a pointer to the specified solution.
   *
   * @throws std::out_of_range if the provided index is out of bounds.
   */
  t_solution* get(unsigned int index) const;

  /**
   * @brief Append a solution at the end of this container.
   *
   * @param s a pointer to the new solution.
   *
   * @throws invalid_argument, if the given solution is not assignable.
   */
  void add(t_solution* s);

  /**
   * @brief Replace a specific solution by a new one in this container.
   *
   * @param index the index of the solution to be removed.
   *
   * @param s a pointer to the new solution.
   *
   * @return a pointer to the replaced solution.
   *
   * @throws invalid_argument, if the given solution is not assignable.
   *
   * @throws std::out_of_range if the provided index is out of bounds.
   */
  t_solution* modify(unsigned int index, t_solution* s);

  /**
   * @brief Remove a specified solution from this container.
   *
   * This method does not delete the specified element, it only remove the
   * element from the container.
   *
   * @param index the index of the solution to be removed.
   *
   * @return the removed element.
   *
   * @throws std::out_of_range if the provided index is out of bounds.
   */
  t_solution* remove(unsigned int index);

  /**
   * @brief Remove and delete each element in this container.
   *
   * This method destroys all solutions in this container and returns a pointer
   * to this container.
   *
   * @return a pointer to this container once is empty.
   */
  set<t_solution>* clean();

  /**
   * @brief Calculate the average fitness of the solutions in this container.
   *
   * @return the average of fitness of the solutions.
   */
  float fitness();

  float fitness_dev();

  /**
   * @brief Sort the solutions according to its fitness value. Better to worse.
   *
   * @param maximization the optimization operation to consider.
   */
  void sort(bool maximization);

  /**
   * @brief Find the best @ref solution in this container depending
   * on the optimization operation.
   *
   * If maximization, the best @ref solution is the one with highest
   * fitness value. If minimization, the best @ref solution is the one with
   * lowest fitness value.
   *
   * @param maximization the optimization operation to consider.
   *
   * @return a pointer to the best @ref solution.
   */
  t_solution* get_best(bool maximization);

  /**
   * @brief Find the best @ref solution in this container depending
   * on the optimization operation.
   *
   * If maximization, the best @ref solution is the one with highest
   * fitness value. If minimization, the best @ref solution is the one with
   * lowest fitness value.
   *
   * @param maximization the optimization operation to consider.
   *
   * @return the index within this container of the best @ref solution.
   */
  int get_best_index(bool maximization);

  /**
   * @brief Find the worst @ref solution optimized by this algorithm depending
   * on the optimization operation.
   *
   * If maximization, the worst @ref solution is the one with lowest
   * fitness value. If minimization, the worst @ref solution is the one with
   * highest fitness value.
   *
   * @param maximization the optimization operation to consider.
   *
   * @return a pointer to the worst @ref solution.
   */
  t_solution* get_worst(bool maximization);

  /**
   * @brief Find the worst @ref solution optimized by this algorithm depending
   * on the optimization operation.
   *
   * If maximization, the worst @ref solution is the one with lowest
   * fitness value. If minimization, the worst @ref solution is the one with
   * highest fitness value.
   *
   * @param maximization the optimization operation to consider.
   *
   * @return the index within this container of the best @ref solution.
   */
  int get_worst_index(bool maximization);

  /**
   * @brief Calculate the average number of function evaluations of the
   * solutions in this container.
   *
   * @return the average of function evaluations.
   */
  int get_evaluations();

  /**
   * @brief Initializes all solutions in this container by calling the
   * @ref init() method of each.
   */
  void generate();

  /**
   * @brief Return a copy to this container.
   *
   * @return a copy of this container.
   */
  set<t_solution>* clone();

  /**
   * Create a copy of this container changing the type of the elements to the
   * desired one.
   *
   * The elements are not copied, only the container. The cast follows the
   * restrictions of a dynamic_cast<t_t_solution*>(t_solution *). User is
   * responsible for deleting the returned container.
   *
   * @return a pointer to the casted container.
   */
  template<class t_t_solution = solution>
  set<t_t_solution>* cast_copy() const;

  virtual ~set();

protected:

  /**
   * @brief The basic constructor for a set.
   *
   * @param size the number of solutions of this container.
   */
  set(unsigned int size = 10);

  /** std::vector containing pointers of the solutions of this container. */
  vector<t_solution*> _solutions;

};

template<class t_solution>
set<t_solution>* set<t_solution>::make(unsigned int size)
{
  return new set<t_solution>(size);
}

template<class t_solution>
int set<t_solution>::size() const
{
  return _solutions.size();
}

template<class t_solution>
int set<t_solution>::get_dim() const
{
  return get(0)->size();
}

template<class t_solution>
t_solution* set<t_solution>::get(unsigned int index) const
{
  if(index >= size())
  {
    throw std::out_of_range("index out of bounds");
  }

  return _solutions[ index ];
}

template<class t_solution>
void set<t_solution>::add(t_solution* s)
{
  if(size() != 0 && get(0)->assignable(s) == false)
  {
    throw invalid_argument("Given solution must be assignable");
  }

  _solutions.push_back( move( s ) );
}

template<class t_solution>
t_solution* set<t_solution>::modify(unsigned int index, t_solution* s)
{
  t_solution* result = get(index);

  if(size() != 0 && get(0)->assignable(s) == false)
  {
    throw invalid_argument("Given solution must be assignable");
  }

  if(index >= size())
  {
    throw std::out_of_range("Index out of bounds");
  }
  _solutions[index] = s;

  return result;
}

template<class t_solution>
t_solution* set<t_solution>::remove(unsigned int index)
{
  if(index >= size())
  {
    throw std::out_of_range("index out of bounds");
  }

  return *_solutions.erase(_solutions.begin() + index);
}

template<class t_solution>
set<t_solution>* set<t_solution>::clean()
{
  for(auto* s : _solutions)
  {
    delete s;
  }

  _solutions.clear();

  return this;
}

template<class t_solution>
float set<t_solution>::fitness()
{
  float ave = 0;

  for(unsigned int i = 0; i < size(); i++)
  {
    ave += get(i)->fitness();
  }

  return ave / size();
}

template<class t_solution>
float set<t_solution>::fitness_dev()
{
  float dev = 0;
  float ave = fitness();

  for(unsigned int i = 0; i < size(); i++)
  {
    dev += pow(ave - get(i)->fitness(), 2);
  }

  return sqrt(dev / (size() - 1));
}

template<class t_solution>
void set<t_solution>::sort(bool maximization)
{
  for(int i = 0; i < size(); i++)
  {
    for(int j = i + 1; j < size(); j++)
    {
      if(get(j)->is_better_than(get(i), maximization))
      {
        auto* aux = _solutions[i];

        _solutions[i] = _solutions[j];
        _solutions[j]  = aux;
      }
    }
  }
}

template<class t_solution>
t_solution* set<t_solution>::get_best(bool maximization)
{
  return get(get_best_index(maximization));
}

template<class t_solution>
int set<t_solution>::get_best_index(bool maximization)
{
  int best = 0;

  for(int i = 1; i < size(); i++)
  {
    if(get(i)->is_better_than(get(best), maximization))
    {
      best = i;
    }
  }

  return best;
}

template<class t_solution>
t_solution* set<t_solution>::get_worst(bool maximization)
{
  return get(get_worst_index(maximization));
}

template<class t_solution>
int set<t_solution>::get_worst_index(bool maximization)
{
  int worst = 0;

  for(int i = 1; i < size(); i++)
  {
    if(get(i)->is_better_than(get(worst), !maximization))
    {
      worst = i;
    }
  }

  return worst;
}

template<class t_solution>
int set<t_solution>::get_evaluations()
{
  int result = 0;

  for(int i = 0; i < size(); i++)
  {
    result += get(i)->get_evaluations();
  }

  return result;
}

template<class t_solution>
void set<t_solution>::generate()
{
  for( int i = 0; i < size(); i++ )
  {
    get(i)->generate();
  }
}

template<class t_solution>
set<t_solution>* set<t_solution>::clone()
{
  auto* result = set<t_solution>::make(size());

  for (int i = 0; i < size(); i++)
  {
    result->add(get(i)->clone());
  }

  return result;
}

template<class t_solution>
template<class t_t_solution>
set<t_t_solution>* set<t_solution>::cast_copy() const
{
  auto* result = set<t_t_solution>::make(size());

  for(int i = 0; i < size(); i++)
  {
    result->add(dynamic_cast<t_t_solution*>(get(i)));
  }

  return result;
}

template<class t_solution>
set<t_solution>::set(unsigned int size)
{
  _solutions.reserve(size);
}

template<class t_solution>
set<t_solution>::~set()
{

}


} // namespace core
} // namespace dnn_opt

#endif
