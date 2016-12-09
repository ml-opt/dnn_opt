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

#ifndef DNN_OPT_CORE_LAYERS_MAX_POOLING_H
#define DNN_OPT_CORE_LAYERS_MAX_POOLING_H

#include <vector>
#include <memory>
#include <math.h>

#include <src/core/layer.h>
#include <src/model/activation_function/identity.h>

using namespace std;
using namespace dnn_opt::core;

namespace dnn_opt
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

    class max_pooling : public layer
    {
    public:

        static shared_ptr< max_pooling > make(  int   in_height,
                                                int   in_width,
                                                int   in_depth,
                                                int   w_height,
                                                int   w_width,
                                                int   stride
                                             )
        {
            return shared_ptr< max_pooling >( new max_pooling( in_height, in_width, in_depth, w_height, w_width, stride ) );
        }

        void propagate( const vector< float >& input, const vector< float >& params, int start, int end ) override
        {
            assert( end - start == 0 );
            assert( input_dimension() == ( int ) input.size() );
            assert( start <= end );
            assert( end <= ( int ) params.size() );
            assert( end - start == count() );

            int  output_index       = 0;

            for( int i = 0; i + _w_height <= _in_height; i += _stride )
            {
                for( int j = 0; j + _w_width <= _in_width; j += _stride )
                {
                    for( int k = 0; k < _in_depth; k++ )
                    {
                        float max_value = input[ get_lineal_index( i, j, k ) ];

                        for( int w_i = i; w_i < i + _w_height; w_i++ )
                        {
                            for( int w_j = j; w_j < j + _w_width; w_j++ )
                            {
                                max_value = max( max_value, input[ get_lineal_index( w_i, w_j, k ) ] );
                            }
                        }

                        _output[ output_index ] = max_value;
                        output_index              += 1;
                    }
                }
            }
        }

        /**
         * @brief count returns the number of parameters that is required by this layer.
         * This is input_dimension() * output_dimension() since this is a fully connected layer.
         *
         * @return the number of parameters.
         */
        int count() const override
        {
            return 0;
        }

        const vector< float >& output() const override
        {
            return _output;
        }

        layer* clone()
        {
            return new max_pooling( _in_height, _in_width, _in_depth, _w_height, _w_width, _stride );
        }

    protected:

        max_pooling( int   in_height,
                     int   in_width,
                     int   in_depth,
                     int   w_height,
                     int   w_width,
                     int   stride
                   )
        : layer    (
                     in_height * in_width * in_depth,
                     in_depth * ( ( in_height - w_height ) / stride + 1 ) * ( ( in_width - w_width ) / stride + 1 ),
                     move( activation_functions::identity::make() )
                   ),
          _output( in_depth * ( ( in_height - w_height ) / stride + 1 ) * ( ( in_width - w_width ) / stride + 1 ) )
        {
            assert( ( in_width - w_width ) % stride == 0 );
            assert( ( in_height - w_height ) % stride == 0 );

            _in_height      = in_height;
            _in_width       = in_width;
            _in_depth       = in_depth;
            _w_height       = w_height;
            _w_width        = w_width;
            _stride         = stride;
        }

    private:

        int get_lineal_index( int x, int y, int z )
        {
            return _in_depth * ( y * _in_width + x ) + z;
        }

        int             _in_height;
        int             _in_width;
        int             _in_depth;
        int             _w_height;
        int             _w_width;
        int             _stride;

        vector< float > _output;
    };
}
}
}

#endif
