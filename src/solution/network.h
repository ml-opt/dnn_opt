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

#ifndef DNN_OPT_CORE_SOLUTIONS_NETWORK
#define DNN_OPT_CORE_SOLUTIONS_NETWORK

#include <memory>
#include <vector>

#include <src/core/sampler.h>
#include <src/core/error_function.h>
#include <src/core/parameter_generator.h>
#include <src/core/solution.h>
#include <src/core/layer.h>

using namespace dnn_opt::core;

namespace dnn_opt
{
namespace core
{
namespace solutions
{
    class network : public solution
    {
    public:

        static unique_ptr< network > make(  bool lazzy, shared_ptr< sampler > s, shared_ptr< error_function > E, shared_ptr< parameter_generator > GEN, int param_count = 10 )
        {
            return unique_ptr< network >( new network( lazzy, s, E, GEN, param_count ) );
        }

        void set( int index, float value ) override
        {
            _params[ index ] = value;

            solution::set( index, value );
        }

        float get( int index ) const override
        {
            return _params[ index];
        }

        int size() const override
        {
            return _params.size();
        }

        vector<float>& get_parameters( ) override
        {
            return _params;
        }

        /**
         * @brief Returns if this solution calculates its fitness in a lazzy way.
         * @return true if fitness is lazzy, false otherwise.
         */
        bool is_lazzy()
        {
            return _lazzy;
        }

        /**
         * @brief Changes the way this solution calculates its fitness.
         * @param lazzy set true for lazzy fitness false for not lazzy.
         */
        void set_lazzy( bool lazzy )
        {
            _modified   = lazzy != _lazzy;
            _lazzy      = lazzy;
        }

        virtual network* clone() override
        {
            network* result             = new network( _lazzy, _s, _E, _generator, _params.size() );

            for( auto& l : _layers )
            {
                result->add_layer( shared_ptr< layer >( l->clone() ) );
            }

            ( *result )                 = ( *this );

            return result;
        }


        virtual bool assignable( solution &s ) const override
        {
            /* Warning: Incomplete method implementation. Check also that contains the same layered structure. */

            return ( typeid( s ) == typeid( network ) ) &&
                   ( size()      == s.size() );
        }

        void add_layer( shared_ptr< layer > l )
        {
            if ( _layers.empty() == false )
            {
                assert( _layers.back()->output_dimension() == l->input_dimension() );
            }

            int count = l->count();

            for( int i = 0; i < count; i++ )
            {
                _params.push_back( _generator->generate() );
            }

            _layers.push_back( l );
        }

        network& operator <<( shared_ptr< layer > l )
        {
            add_layer( l );

            return *this;
        }

    protected:

        network( bool lazzy, shared_ptr< sampler > s, shared_ptr< error_function > E, shared_ptr< parameter_generator > generator, int param_count = 10 )
            : solution( generator ), _s( s ), _E( E ), _lazzy( lazzy )
        {
            _params.reserve( param_count );
        }

        float calculate_fitness() override
        {
            float loss = 0;

            if( _lazzy == true )
            {
                loss = error( _s->sample_input(),  _s->sample_output() );
            }
            else
            {
                loss = error( _s->input(),  _s->output() );
            }

            return loss;
        }

        virtual float error( const vector< vector< float > >& input, const vector< vector< float > >& expected )
        {
            float                       error = 0;
            vector< vector< float > >   real;

            real.reserve( input.size() );

            for( const auto &pattern : input )
            {
                real.push_back( predict( pattern ) );
            }

            error = _E->error( real, expected );

            return error;
        }

        vector< float > predict( const vector< float > &in )
        {
            int    start   = 0;
            int    end     = _layers.front()->count();

            _layers.front()->propagate( in, _params, start, end );

            for( unsigned int i = 1; i < _layers.size(); i++ )
            {
                start      = end;
                end        = start + _layers[ i ]->count();

                _layers[ i ]->propagate( _layers[ i - 1 ]->output(), _params, start, end  );
            }

            return _layers.back()->output();
        }

        vector< float >                     _params;
        vector< shared_ptr < layer > >      _layers;

        shared_ptr< sampler >               _s;

        shared_ptr< error_function >        _E;

        bool                                _lazzy;
    };
}
}
}

#endif

