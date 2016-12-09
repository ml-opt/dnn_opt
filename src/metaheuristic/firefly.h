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

#ifndef DNN_OPT_CORE_ALGORITHMS_FIREFLY_H
#define DNN_OPT_CORE_ALGORITHMS_FIREFLY_H

#include <memory>
#include <math.h>
#include <random>
#include <cassert>

#include <src/core/algorithm.h>
#include <src/core/solution_set.h>

using namespace std;
using namespace dnn_opt;

namespace dnn_opt
{
namespace core
{
namespace algorithms
{
    /**
     * @brief The firefly class implements an optimization metaheuristic algorithm called Firefly Algorithm (FA).
     * This is a population based algorithm inspired in the bio-luminicence of fireflies.
     *
     * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @date July, 2016
     * @version 1.0
     */
    class firefly : public algorithm
    {
    public:

        /**
         * @brief A function that implements factory pattern for this optimization problem. Returns a new
         * parametrized instance of a Firefly Algorithm over a set of candidate solutions to optimize.
         *
         * @param light_decay the absorption of light by the space.
         * @param rand_influence the influence of randomess in the problem.
         * @param init_bright the bright of a firefly light when distance is cero.
         * @param solutions a set of candidate solutions to optimize.
         * @param maximization if true this is a maximization problem, if false a minimization problem. Default is true.
         *
         * @return an `unique_ptr< firefly >` with a reference to an instance of this class.
         */
        static unique_ptr< firefly > make( float light_decay, float rand_influence, float init_bright, unique_ptr< solution_set > solutions, bool maximization = true )
        {
            return  unique_ptr< firefly >( new firefly( light_decay, rand_influence, init_bright, std::move( solutions ), maximization ) );
        }

        /**
         * @brief Performs a single steep of optimization for this algorithm.
         */
        void optimize() override
        {
            int population_size = _solutions->size();

            _solutions->sort( _maximization == false );

            for( int i = 0; i < population_size; i++ )
            {
                for( int j = 0; j < population_size; j++ )
                {
                    if( _maximization == true && _solutions->get( i )->fitness() < _solutions->get( j )->fitness() )
                    {
                        move( i, j );
                    }

                    if( _maximization == false && _solutions->get( i )->fitness() > _solutions->get( j )->fitness() )
                    {
                        move( i, j );
                    }
                }
            }

            _iteration += 1;
        }

        using algorithm::optimize;

    protected:

        /**
         * @brief Creates a new instance of the Firefly Algorithm. Derived classes should implement factory
         * pattern.
         *
         * @param light_decay the absorption of light by the space.
         * @param rand_influence the influence of randomess in the problem.
         * @param init_bright the bright of a firefly light when distance is cero.
         * @param solutions a set of candidate solutions to optimize.
         * @param maximization if true this is a maximization problem, if false a minimization problem. Default is true.
         */
        firefly( float light_decay, float rand_influence, float init_bright, unique_ptr< solution_set > solutions, bool maximization = true )
            : algorithm( std::move( solutions ) )
        {
            random_device device;

            _generator              = new default_random_engine( device() );
            _distribution           = new uniform_real_distribution< >( 0, 1 );
            _maximization           = maximization;

            _light_decay            = light_decay;
            _rand_influence         = rand_influence;
            _init_bright            = init_bright;
            _iteration              = 0;
        }

    private:

        /**
         * @brief Move a given firefly to another.
         *
         * @param source the firefly who is going to move.
         * @param target the destination or direction in what the source firefly will move.
         *
         * @throws assertion if:
         *      - the soruce firefly is the same that the target firefly, `source == target`
         *      - the source and the target firefly are not valid index in the solution_set of
         *        this algorithm.
         */
        void move( int source, int target )
        {
            /* source firefly must be different to target firefly */
            assert( source != target );

            /* source and target firefly must be in the solution_set range */
            assert( source >= 0 && source < _solutions->size() );
            assert( target >= 0 && target < _solutions->size() );

            float   attraction       =  _init_bright / ( 1 + _light_decay * pow( distance( source, target ), 2 ) );
            int     parameter_count  = _solutions->get(0)->size();
            float   r                = _rand_influence * pow( 0.95, _iteration );

            for( int i = 0; i < parameter_count; i++ )
            {
                float source_param = _solutions->get( source )->get( i );
                float target_param = _solutions->get( target )->get( i );
                float random       = r * ( next_random() - 0.5 );

                _solutions->get( source )->set( i, source_param + attraction * ( target_param - source_param ) + random );
            }
        }

        /**
         * @brief The Eucliden distance between two fireflies.
         *
         * @param source the first firefly.
         * @param target the second firefly.
         *
         * @return the euclidean distance between source firefly and target firefly.
         *
         * @throws assertion if:
         *      - the soruce firefly is the same that the target firefly, `source == target`
         *      - the source and the target firefly are not valid index in the solution_set of
         *        this algorithm.
         */
        double distance( int source, int target )
        {
            /* source firefly must be different to target firefly */
            assert( source != target );

            /* source and target firefly must be in the solution_set range */
            assert( source >= 0 && source < _solutions->size() );
            assert( target >= 0 && target < _solutions->size() );

            float distance         = 0;
            int parameter_count    = _solutions->get(0)->size();

            for ( int i = 0; i < parameter_count; i++ )
            {
                double source_param = _solutions->get( source )->get( i );
                double target_param = _solutions->get( target )->get( i );

                distance += pow( target_param - source_param, 2 );
            }

            return sqrt( distance );
        }

        /**
         * @brief Returns a normal distributed random number with mean = 0 and std. deviation = 0.5.
         * @return a random number.
         */
        float next_random()
        {
            return (* _distribution)( *_generator );
        }

        float                          _light_decay;
        float                          _rand_influence;
        float                          _init_bright;
        int                            _iteration;

        bool                           _maximization;

        default_random_engine*         _generator;
        uniform_real_distribution< >*  _distribution;
    };
}
}
}

#endif
