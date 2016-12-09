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

#ifndef DNN_OPT_CORE_ALGORITHMS_GRAY_WOLF_H
#define DNN_OPT_CORE_ALGORITHMS_GRAY_WOLF_H

#include <memory>
#include <random>

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
     * @brief The gray_wolf class implements an optimization metaheuristic algorithm called Gray Wolf
     * Optimizer (GWO). This is a population based algorithm inspired in the hunting procedure of gray
     * wolfs.
     *
     * ## References
     *      - MIRJALILI, Seyedali; MIRJALILI, Seyed Mohammad; LEWIS, Andrew. Grey wolf optimizer. Advances
     *        in Engineering Software, 2014, vol. 69, p. 46-61.
     *
     * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @date July, 2016
     * @version 1.0
     */
    class gray_wolf : public algorithm
    {
    public:

        /**
         * @brief A function that implements factory pattern for this optimization problem. Returns a new
         * parametrized instance of a gray_wolf class over a set of candidate solutions to optimize.
         *
         * @param decrease_factor   a parameters that controls the `a` parameter of the original algorithm.
         *                          This parameter have to be tunned according to the expected number of
         *                          optimization steeps.
         * @param solutions a set of individuals.
         *
         * @return an instance of a gray_wolf class.
         */
        static unique_ptr< gray_wolf > make( float decrease_factor, unique_ptr< solution_set > solutions )
        {
            return unique_ptr< gray_wolf >( new gray_wolf( decrease_factor, std::move( solutions ) ) );
        }

        /**
         * @brief Performs a single steep of optimization for this algorithm.
         */
        void optimize() override
        {
            vector < double > R1;
            vector < double > R2;

            R1.reserve( _dimension );
            R2.reserve( _dimension );

            for( int i = 0; i < _dimension; i++ )
            {
                R1.push_back( next_random() );
                R2.push_back( next_random() );
            }

            /* Check decrease factor and _a parameter */

            _a -= _decrease_factor;

            /* Fix this to locate only the 3 best solutions */

            _solutions->sort();

            for( int i = 0; i < _solutions->size(); i++ )
            {
                for( int j = 0; j < _dimension; j++ )
                {
                    float A = 2 * _a * R1[ j ] - _a;
                    float C = 2 * R2[ j ];

                    float D = fabs( C * _solutions->get( _solutions->size() - 1 )->get( j ) - _solutions->get( i )->get( j ) );
                    float X1 = _solutions->get( _solutions->size() - 1 )->get( j ) - A * D;

                    A = 2 * _a * R1[ j ] - _a;
                    C = 2 * R2[ j ];

                    D = fabs( C * _solutions->get( _solutions->size() - 2 )->get( j ) - _solutions->get( i )->get( j ) );
                    float X2 = _solutions->get( _solutions->size() - 2 )->get( j ) - A * D;

                    A = 2 * _a * R1[j] - _a;
                    C = 2 * R2[j];

                    D = fabs( C * _solutions->get( _solutions->size() - 3 )->get( j ) - _solutions->get( i )->get( j ) );
                    float X3 = _solutions->get( _solutions->size() - 3 )->get( j ) - A * D;

                    _solutions->get( i )->set( j, ( X1 + X2 + X3 ) / 3 );
                }
            }
        }

        using algorithm::optimize;

    protected:

        /**
         * @brief The basic contructor for a gray_wolf optimization class.
         *
         * @param decrease_factor   a parameters that controls the <a> parameter of the original algorithm.
         *                          This parameter have to be tunned according to the expected number of
         *                          optimization steeps.
         * @param solutions a set of individuals.
         *
         * @throws invalid_argument, if the given solution_set does not have at least one solution to optimize.
         */
        gray_wolf( float decrease_factor, unique_ptr< solution_set > solutions ) : algorithm( std::move( solutions ) )
        {
            random_device device;

            _decrease_factor        = decrease_factor;

            _generator              = new default_random_engine( device() );
            _normal_distribution    = new normal_distribution< float >( 0.5, 0.5 );
            _selector               = new uniform_int_distribution<>( 0, _solutions->size() - 1 );

            _dimension              = _solutions->get( 0 )->size();
        }

    private:

        /**
         * @brief Returns a normal distributed random number with mean = 0 and std. deviation = 0.5.
         * @return a random number.
         */
        float next_random()
        {
            return (* _normal_distribution)( *_generator );
        }


        float                          _decrease_factor;
        int                            _dimension;
        float                          _a;

        default_random_engine*         _generator;
        normal_distribution< float >*  _normal_distribution;
        uniform_int_distribution<>*    _selector;
    };
}
}
}

#endif
