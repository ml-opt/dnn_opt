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

#ifndef DNN_OPT_CORE_ALGORITHMS_PSO_H
#define DNN_OPT_CORE_ALGORITHMS_PSO_H

#include <memory>

#include <src/core/algorithm.h>

using namespace std;
using namespace dnn_opt;

namespace dnn_opt
{
namespace core
{
namespace algorithms
{
    /**
     * @brief The pso class implements an optimization metaheuristic algorithm called Particle Swarm
     * Optimization (PSO). This is a population based algorithm inspired in the movements of swarms.
     *
     * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @version 1.0
     * @date July, 2016
     */
    class pso : public algorithm
    {
    public:

        /**
         * @brief A function that implements factory pattern for this optimization problem. Returns a new
         * parametrized instance of a pso class over a set of candidate solutions to optimize.
         *
         * @param local_param the contribution of local best solutions.
         * @param global_param the contribution of the global best solution.
         * @param speed_param the contribution of the speed in each particle movement.
         * @param solutions a set of individuals.
         *
         * @return a parametrized instance of a pso class.
         */
        static unique_ptr< pso > make( float local_param, float global_param, float speed_param, unique_ptr< solution_set > solutions, bool maximization = true )
        {
            return unique_ptr< pso >( new pso( local_param, global_param, speed_param, std::move( solutions ), maximization ) );
        }

        /**
         * @brief Performs a single steep of optimization for this algorithm.
         */
        void optimize()
        {
            int population_size = _solutions->size();

            for( int i = 0; i < population_size; i++ )
            {
                update_speed( i );
                update_position( i );
                update_local( i );                
            }
        }

        using algorithm::optimize;

    protected:

        /**
         * @brief The basic contructor of a pso class.
         *
         * @param local_param the contribution of local best solutions.
         * @param global_param the contribution of the global best solution.
         * @param speed_param the contribution of the speed in each particle movement.
         * @param solutions a set of individuals.
         */
        pso( float local_param, float global_param, float speed_param, unique_ptr< solution_set > solutions, bool maximization = true )
            : algorithm( std::move( solutions ) )
        {
            random_device device;

            _generator              = new default_random_engine( device() );
            _uniform_distribution   = new uniform_real_distribution<>( 0, 1 );

            _local_param            = local_param;
            _global_param           = global_param;
            _speed_param            = speed_param;
            _global_solution        = 0;
            _maximization           = maximization;

             _local_solutions       = std::move( _solutions->clone() );
             _speed                 = std::move( _solutions->clone() );

            _speed->init();

            for( int i = 1; i < _local_solutions->size(); i++ )
            {
                update_global( i );
            }
        }

    private:

        /**
         * @brief Update the speed of a given solution.
         * @param index the index of the solution to be updated.
         */
        void update_speed( int index )
        {
            int size = _speed->get( index )->size();

            for( int i = 0; i < size; i++ )
            {
                float current_value    = _speed_param * _speed->get( index )->get( i ) ;
                float local_value      = _local_param * ( *_uniform_distribution )( *_generator ) * ( _local_solutions->get( index )->get( i ) - _solutions->get( index )->get( i ) );
                float global_value     = _global_param * ( *_uniform_distribution )( *_generator ) * ( _local_solutions->get( _global_param )->get( i ) - _solutions->get( index )->get( i ) );

                _speed->get( index )->set( i, current_value + local_value + global_value );
            }
        }

        /**
         * @brief Update the position of a fiven solution.assignable()
         * @param index the index of the solution to be updated.
         */
        void update_position( int index )
        {
            int size = _solutions->get( index )->size();

            for( int i = 0; i < size; i++ )
            {
                _solutions->get( index )->set( i, _solutions->get( index )->get( i ) + _speed->get( index )->get( i ) );
            }
        }

        /**
         * @brief Update the local best solution of a given solution if the current solution is better than
         * the current best local solution.
         * @param index the index of the solution to be updated.
         */
        void update_local( int index )
        {
            if( _maximization == true && _local_solutions->get( index )->fitness() < _solutions->get( index )->fitness() )
            {
                _local_solutions->set( index, std::move( unique_ptr< solution >( _solutions->get( index )->clone() ) ) );

                update_global( index );
            } else if( _maximization == false && _local_solutions->get( index )->fitness() > _solutions->get( index )->fitness() )
            {
                _local_solutions->set( index, std::move( unique_ptr< solution >( _solutions->get( index )->clone() ) ) );

                update_global( index );
            }
        }

        /**
         * @brief Update the global solution if a given solution is better than the current best global solution.
         * @param index the index of the solution to be updated.
         */
        void update_global( int index )
        {
            if( _maximization == true && _local_solutions->get( _global_solution )->fitness() < _local_solutions->get( index )->fitness() )
            {
                _global_solution = index;
            } else if( _maximization == false && _local_solutions->get( _global_solution )->fitness() > _local_solutions->get( index )->fitness() )
            {
                _global_solution = index;
            }
        }

        float                                  _local_param;
        float                                  _global_param;
        float                                  _speed_param;
        bool                                   _maximization;

        unique_ptr< solution_set >             _local_solutions;
        unique_ptr< solution_set >             _speed;
        int                                    _global_solution;

        default_random_engine*                 _generator;
        uniform_real_distribution<>*           _uniform_distribution;
    };
}
}
}

#endif
