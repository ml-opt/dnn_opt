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

#ifndef DNN_OPT_CORE_ALGORITHMS_CUCKOO_H
#define DNN_OPT_CORE_ALGORITHMS_CUCKOO_H

#include <memory>
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
     * @brief The cuckoo class implements an optimization metaheuristic algorithm called Cuckoo Search (CS).
     * This is a population based algorithm equiped with levy flights that allows an improved explotation and
     * exploration of the search space.
     *
     * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @date July, 2016
     * @version 1.0
     */
    class cuckoo : public algorithm
    {
    public:

        /**
         * @brief A function that implements factory pattern for this optimization problem. Returns a new
         * parametrized instance of a Cuckoo Search algorithm over a set of candidate solutions to optimize.

         * @param scale is the scale of the search space.
         * @param levy is the Levy parameter for the levy's walk.
         * @param replacement is the proportion to replace bad solutions.
         * @param solutions is the set of candidate solutions.
         *
         * @return an `unique_ptr< cuckoo >` with a reference to an instance of this class.
         */
        static unique_ptr< cuckoo > make( float scale, float levy, float replacement, unique_ptr< solution_set > solutions, bool maximization = true )
        {
            return unique_ptr< cuckoo >( new cuckoo( scale, levy, replacement, std::move( solutions ), maximization ) );
        }

        /**
         * @brief Performs a single steep of optimization for this algorithm.
         */
        void optimize() override
        {
            for( int i = 0; i < _solutions->size(); i++ )
            {
                update_global( i );
            }

            int source_index        = ( *_selector )( *_generator );
            int target_index        = ( *_selector )( *_generator );

            while( source_index == target_index )
            {
                target_index        = ( *_selector )( *_generator );
            }

            auto &target        = _solutions->get( target_index );

            unique_ptr< solution > updated( generate_new_cuckoo( source_index ) );

            if( updated->fitness() > target->fitness() )
            {
                _solutions->set( target_index, std::move( updated ) );
            }

            _solutions->sort( _maximization );

            int replace_count = _solutions->size() * _replacement;

            for( int i = 0; i < replace_count; i++ )
            {
                _solutions->get( i )->init();
            }
        }

        using algorithm::optimize;

    protected:

        /**
         * @brief Creates a new instance of the Cuckoo Search algorithm. This constructor is protector
         * and derived classes should implement a factory pattern.
         *
         * @param scale is the scale of the search space.
         * @param levy is the Levy parameter for the levy's walk.
         * @param replacement is the proportion to replace bad solutions.
         * @param solutions is the set of candidate solutions.
         */
        cuckoo( float scale, float levy, float replacement, unique_ptr< solution_set > solutions, bool maximization )
            : algorithm( std::move( solutions ) )
        {
            random_device device;

            _scale                  = scale;
            _levy                   = levy;
            _replacement            = replacement;
            _global_solution        = 0;
            _maximization           = maximization;

            _omega                  = pow( ( tgamma( 1 + levy ) * sin( 3.14159265f * levy / 2 ) ) / ( tgamma( ( 1 + levy ) / 2 ) * levy * pow( 2, (levy - 1 ) / 2 ) ) , 1 / levy);

            _generator              = new default_random_engine( device() );
            _normal_distribution    = new normal_distribution< float >( 0, 0.5 );
            _selector               = new uniform_int_distribution<>( 0, _solutions->size() - 1 );
        }

    private:

        /**
         * @brief Generates a new candidate solution by performing a levy flight from a given
         * solution.
         *
         * @param index the index of a solution from who a levy flight will be performed.
         *
         * @return a pointer to the new generated solution.
         *
         * @throws assertion if the given index is not contained in the solution_set of this class.
         */
        solution* generate_new_cuckoo( int index )
        {
            /* index must be in the solution_set range */

            assert( index >= 0 && index < _solutions->size() );

            solution* result = _solutions->get( index )->clone();

            float v    = (* _normal_distribution)( *_generator ) + 0.5;
            float u    = (* _normal_distribution)( *_generator ) + ( _omega - 0.5 );
            float levy = u  / pow( fabs( v ), 1 / _levy );

            int size    = _solutions->get( index )->size();

            for( int i = 0; i < size; i++ )
            {
                result->set( i, _solutions->get( index )->get( i ) + _scale * levy * _solutions->get( _global_solution )->get( i ) * ( *_normal_distribution )( *_generator ) );
            }

            return result;
        }

        /**
         * @brief Update the index of the best solution if the given solution is better that the current best.
         * @param index the index of a solution.
         *
         * @throws assertion if the given index is not contained in the solution_set of this class.
         */
        void update_global( int index )
        {
            /* index must be in the solution_set range */

            assert( index >= 0 && index < _solutions->size() );

            if( _solutions->get( _global_solution )->fitness() > _solutions->get( index )->fitness() )
            {
                _global_solution = index;
            }
        }

        float                          _scale;
        float                          _levy;
        float                          _replacement;

        float                          _omega;
        int                            _global_solution;
        bool                           _maximization;

        default_random_engine*         _generator;
        normal_distribution< float >*  _normal_distribution;
        uniform_int_distribution<>*    _selector;
    };
}
}
}

#endif
