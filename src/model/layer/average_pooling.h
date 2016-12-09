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

#ifndef DNN_OPT_CORE_LAYERS_AVERAGE_POOLING_H
#define DNN_OPT_CORE_LAYERS_AVERAGE_POOLING_H

#include <vector>
#include <memory>
#include <math.h>

#include <src/core/layer.h>
#include <src/model/activation_function/identity.h>

using namespace std;

namespace ann_pt
{
namespace core
{
namespace layers
{
    /**
     *
     * @author: Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @date October, 2016
     * @version 1.0
     */

    class average_pooling : public layer
    {
    public:

        static shared_ptr< fully_connected > make( int                                in_dim,
                                                   int                                out_dim,
                                                   unique_ptr< activation_function >  AF
                                                 )
        {
            return shared_ptr< fully_connected >( new fully_connected( in_dim, out_dim, move( AF ) ) );
        }

        vector< float > propagate( const vector< float >&   input,
                                   const vector< float >&   params,
                                   unsigned int             start,
                                   unsigned int             end
                                 ) override
        {

        }

        /**
         * @brief count returns the number of parameters that is required by this layer.
         * This is input_dimension() * output_dimension() since this is a fully connected layer.
         *
         * @return the number of parameters.
         */
        unsigned int count() const override
        {
            return 0;
        }

    protected:


        average_pooling( unsigned int   in_height,
                         unsigned int   in_width,
                         unsigned int   in_depth,
                         unsigned int   w_height,
                         unsigned int   w_width,
                         unsigned int   stride
                       )
        : layer        (
                         in_height * in_width * in_depth,
                         in_depth * ( ( in_height - w_height ) / stride + 1 ) * ( ( in_width - w_width ) / stride + 1 ),
                         move( activation::identity::make() )
                       )
        {

        }

    private:

        int get_lineal_index( int x, int y, int z )
        {
            return _in_depth * ( y * _in_width + x ) + z;
        }

        unsigned int    _in_height;
        unsigned int    _in_width;
        unsigned int    _in_depth;
        unsigned int    _w_height;
        unsigned int    _w_width;
        unsigned int    _stride;
    };
}
}
}

#endif
