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

#ifndef DNN_OPT_CORE_ALGORITHM_H
#define DNN_OPT_CORE_ALGORITHM_H

#include <memory>
#include <iostream>

#include <src/core/solution_set.h>

using namespace std;
using namespace dnn_opt;

namespace dnn_opt
{
namespace core
{
    /**
     * This class represents an abstract optimization algorithm capable of
     * define the basic functionalities of any meta-heuristic. In order to extend
     * the library, new algorithms shuold derive from this class.
     *
     * Derived classes are free to introduce custom algorithm parameters on its
     * constructor but always have to provide a solution_set class in order to define
     * the individuals to be optimized. In case that the new algorithm is not population
     * based, it can be used a solution_set class with only one individual.
     *
     * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @date June, 2016
     * @version 1.0
     */
    class algorithm
    {
    public:

        /**
         * @brief Abstract method that implements a single steep of optimization.
         */
        virtual void optimize() = 0;

        /**
         * @brief This method performs multiple steeps of optimization of this optimization
         * algorithm.
         *
         * @param count is the number of steeps to perform.
         */
        void optimize( int count )
        {
            for( int i = 0; i < count; i++ )
            {
                optimize();
            }
        }

        /**
         * @brief Returns the solution_set used by this algorithm.
         *
         * @return an unique_ptr pointing to an instance of the solution set used by this
         * algorithm.
         */
        unique_ptr < solution_set >& get_solutions()
        {
            return _solutions;
        }

    protected:

        /**
         * @brief The basic contructor for an optimization algorithm.
         * In case that the new algorithm is not population based we
         * strongly recommend to use a solution_set class with only one individual.
         *
         * @param solutions is the set of individuals to optimize.
         */
        algorithm( unique_ptr < solution_set > solutions ) : _solutions( std::move( solutions ) )
        {

        }

        unique_ptr < solution_set >        _solutions;
    };
}
}

#endif
