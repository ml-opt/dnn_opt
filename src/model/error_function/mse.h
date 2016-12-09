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

#ifndef DNN_OPT_CORE_ERRORS_MSE
#define DNN_OPT_CORE_ERRORS__MSE

#include <cassert>
#include <vector>
#include <math.h>
#include <memory>

#include <src/core/error_function.h>

using namespace std;
using namespace dnn_opt::core;

namespace dnn_opt
{
namespace core
{
namespace error_functions
{
    class mse : public error_function
    {
    public:

        static shared_ptr< mse > make()
        {
            return shared_ptr< mse >( new mse() );
        }

        float error( const vector< vector < float > > &real, const vector< vector < float > > &expected ) override
        {
            error_function::error( real, expected );

            float result = 0;

            for( unsigned int i = 0; i < real.size(); i++ )
            {
                float squared_sum = 0;

                const vector< float >& real_pattern       = real[ i ];
                const vector< float >& expected_pattern   = expected[ i ];

                assert( real_pattern.size() == expected_pattern.size() );

                for( unsigned int j = 0; j < real_pattern.size(); j++ )
                {
                    squared_sum += pow( expected_pattern[ j ] - real_pattern[ j ], 2 );
                }

                result += squared_sum / real_pattern.size();
            }

            return result / real.size();
        }
    };
}
}
}

#endif

