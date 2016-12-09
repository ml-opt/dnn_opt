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

#ifndef DNN_OPT_CORE_SAMPLER_H
#define DNN_OPT_CORE_SAMPLER_H

#include <cassert>
#include <vector>
#include <random>
#include <memory>

using namespace std;

namespace dnn_opt
{
namespace core
{
    /**
     * @brief The sampler class is intended to maintain samples of training data for neural networks
     * that is going to be used while optimization. Basically contains getters methods of input
     * and output training patterns and getters of input and output training patterns of a random
     * subset of the original data. A training pattern is a `vector< float >.`
     *
     * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @date June, 2016
     * @version 1.0
     */
    class sampler
    {
    public:

        /**
         * @brief Returns a shared_ptr of a sampler instance.
         *
         * @param size the number of training patterns to include in the random subset.
         * @param input the full set of input training patterns.
         * @param output the full set of output training patterns.
         */
        static shared_ptr< sampler > make( int size, const vector< vector< float > > &input, const vector< vector< float > > &output )
        {


            return shared_ptr< sampler >( new sampler( size, input, output ) );
        }

        /**
         * @brief Returns the original set of input training patterns.
         * @return a vector with the input training patterns.
         */
        const vector< vector< float > >& input() const
        {
            return _input;
        }

        /**
         * @brief Returns the original set of output training patterns.
         * @return a vector with the output training patterns.
         */
        const vector< vector< float > >& output() const
        {
            return _output;
        }

        /**
         * @brief Returns a random subset of input training patterns.
         * @return a vector with the input training patterns.
         */
        const vector< vector< float > >& sample_input()
        {
            return _sample_input;
        }

        /**
         * @brief Returns the random subset of output training patterns.
         * @return a vector with the output training patterns.
         */
        const vector< vector< float > >& sample_output() const
        {
            return _sample_output;
        }

        /**
         * @brief Returns the number of training patterns in the original dataset.
         * @return the number of training patterns.
         */
        int size() const
        {
            return _input.size();
        }

        /**
         * @brief Returns the number of training patterns in the random subset.
         * @return the number of training patterns.
         */
        int sample_size() const
        {
            return _size;
        }

        /**
         * @brief sample forces the re-generation of the random sample.
         */
        void sample( )
        {
            random_device               device;
            default_random_engine       generator( device() );
            uniform_int_distribution<>  selector( 0, _input.size() );

            _sample_input.clear();
            _sample_output.clear();

            for( int i = 0; i < _size; i++ )
            {

                int selection = selector( generator );

                _sample_input.push_back( _input[ selection ] );
                _sample_output.push_back( _output[ selection ] );
            }
        }

    protected:

        /**
         * @brief Creates a sampler object with a random subset with the given number of training
         * patterns from the original training data.
         *
         * @param size the number of training patterns to include in the random subset.
         * @param input the full set of input training patterns.
         * @param output the full set of output training patterns.
         *
         * @throws assertion if input and output does not have the same size or if the size of the
         * subset is bigger than the number of training patterns.
         */
        sampler( int size, const vector< vector< float > > &input, const vector< vector< float > > &output )
        {
            assert( input.size() == output.size() );
            assert( size < input.size() );

            _input      = input;
            _output     = output;
            _size       = size;

            sample( );
        }

    protected:

        vector< vector< float > >   _input;
        vector< vector< float > >   _output;

        vector< vector< float > >   _sample_input;
        vector< vector< float > >   _sample_output;

        int                         _size;
    };
}
}

#endif
