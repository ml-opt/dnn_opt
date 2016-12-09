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

#ifndef DNN_OPT_CORE_ACTIVATION_SOFTMAX
#define DNN_OPT_CORE_ACTIVATION_SOFTMAX

#include <math.h>
#include <algorithm>
#include <memory>

#include <src/core/activation_function.h>

using namespace std;
using namespace dnn_opt::core;

namespace dnn_opt
{
namespace core
{
namespace activation_functions
{
    /**
     * @brief The softmax class represents a softmax `  ` function that can be used
     * by an artificial neural network as activation function.
     *
     * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @date September, 2016
     * @version 1.0
     */
    class softmax : public activation_function
    {
    public:

        static shared_ptr< softmax > make()
        {
            return shared_ptr< elu >( new softmax() );
        }

        float activation( const vector< float > &output, int value ) override
        {
            float result    = 0;
            float alpha     = max_element( output.begin(), output.end() );
            float numer     = exp( output[ value ] - alpha );
            float denom     = 0;

            for( auto& out : output)
            {
                denom      += exp( out - alpha );
            }

            result          = numer /denom;

            return result;
        }
    };
}
}
}

#endif

