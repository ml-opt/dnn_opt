/*
    Copyright (c) 2016, Jairo Rojas-Delgado
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

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef DNN_OPT_CORE_SOLUTION_SET_H
#define DNN_OPT_CORE_SOLUTION_SET_H

#include <memory>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <src/core/solution.h>

using namespace std;
using namespace dnn_opt;

namespace dnn_opt
{
namespace core
{
    /**
     * @brief The solution_set class is intended to manage a set of optimization solutions for
     * a determined optimization problem. This a helper class that can be usefull for population
     * based optimization metaheuristics.
     *
     * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @date June, 2016
     * @version 1.0
     */
    class solution_set
    {
    public:

        /**
         * @brief `make` creates a new instance of a solution_set class. This is a an implementation
         * of the factory pattern.
         *
         * @param size the number of solutions that this solution_set is going to manage. The solution_set
         * can grow beyond this limit but settinng it at first can save time from re-allocation of memory.
         * Default is 10.
         *
         * @return an `unique_ptr< solution_set >` poining to a new instance of this class.
         */
        static unique_ptr< solution_set > make( int size = 10 )
        {
            return unique_ptr< solution_set >( new solution_set( size ) );
        }

        /**
         * @brief Returns the number of solutions of this container.
         * @return the number of solutions of this container.
         */
        int size()
        {
            return _solutions.size();
        }

        /**
         * @brief Returns an `unique_ptr< solution >` to a specified solution in this container.
         *
         * @param index the index of a solution in this container.
         *
         * @return an `unique_ptr< solution >` to the specified solution.
         */
        unique_ptr< solution >& get( int index )
        {
            assert( index >= 0 && index < size() );

            return _solutions[ index ];
        }

        /**
         * @brief Appends a solution to the end of this container.
         *
         * @param s a `unique_ptr< solution >` that references to the solution.
         *
         * @throws assertion, if the given solution is not assignable to the others.
         */
        void add( unique_ptr< solution > s )
        {
            assert( size() == 0 || _solutions[0]->assignable( *s ) == true );

            _solutions.push_back( move( s ) );
        }

        /**
         * @brief Changes a specific solution by a new one in this container.
         *
         * @param index the index of the solution to be removed.
         * @param s a unique pointer that references to the new solution.
         *
         * @throws assertion, if the given solution is not assignable to the others.
         */
        void set( int index, unique_ptr< solution > s  )
        {
            assert( index >= 0 && index < size() );
            assert( size() == 0 || _solutions[0]->assignable( *s ) == true );

            _solutions[ index ] = std::move( s );
        }

        /**
         * @brief Removes a specified solution from this container.
         * @param index the index of the solution to be removed.
         */
        void remove( int index)
        {
            assert( index >= 0 && index < size() );

            _solutions.erase( _solutions.begin() + index );
        }

        /**
         * @brief Calculates the average fitness of the solutions in this container.
         * @return the average of fitness of the solutions.
         */
        float fitness()
        {
            float ave = 0;

            for( unsigned int i = 0; i < _solutions.size(); i++ )
            {
                ave += _solutions[ i ]->fitness();
            }

            return ave / _solutions.size();
        }

        /**
         * @brief Sorts the solutions of this container by its fitness.
         *
         * @param lower_higher determines the ordering creiteria, `true` stands for lower to higher order and
         * `false` stands for higher to lower order.
         */
        void sort( bool lower_higher )
        {
            std::sort( _solutions.begin(), _solutions.end(), [ &lower_higher ]( unique_ptr< solution >& first, unique_ptr< solution >& second )
            {
                if( lower_higher == true )
                {
                    return first->fitness() < second->fitness();
                } else
                {
                    return first->fitness() > second->fitness();
                }
            } );
        }

        /**
         * @brief Initializes all solutions of this container by calling the init method of each of them.
         */
        void init()
        {
            for( int i = 0; i < size(); i++ )
            {
                 _solutions[ i ]->init();
            }
        }

        /**
         * @brief Returns a copy to this container.
         * @return an `unique_ptr< solution_set >` which is a copy of this container.
         */
        unique_ptr< solution_set > clone()
        {
            auto copy = make( size() );

            for( auto &s : _solutions )
            {
                copy->add( move( unique_ptr< solution >( s->clone() ) ) );
            }

            return copy;
        }

    protected:

        /**
         * @brief The basic constructor for a solution_set.
         * @param size the number of solutions that this solution_set is going to manage. The solution_set
         * can grow beyond this limit but settinng it at first can save time from re-allocation of memory.
         */
        solution_set( int size = 10 )
        {
            _solutions.reserve( size );
        }

    private:

        /** This vector contains all the solutions of this container. */
        vector< unique_ptr< solution > >    _solutions;
    };
}
}

#endif
