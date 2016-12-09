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

#ifndef DNN_OPT_CORE_SOLUTIONS_DE_JUNG
#define DNN_OPT_CORE_SOLUTIONS_DE_JUNG

#include <memory>
#include <vector>
#include <math.h>

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

    /**
     * @brief The de_jung class represents an optimization solutions which fitness cost is calculated via
     * De' Jung function.
     *
     * De'Jung function have a global minima in {0,..., 0} with a value of 0. A commonly used search domain
     * for testing is [-5, 5].
     *
     * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @version 1.0
     * @date November, 2016
     */
    class de_jung : public solution
    {
    public:

        /**
         * @brief Returns an `unique_ptr< de_jung >` with a reference to an instance of this object. This method
         * is an implementation of the factory pattern.
         *
         * @param generator a shared pointer to an instance of a paramter_generator class. The parameter_generator
         * is used to populate the parameters of this solution.
         *
         * @param param_count is the number of paramters for this solution. Default is 10.
         *
         * @return an unique_ptr poining to an instance of de_jung class.
         */
        static unique_ptr< de_jung > make( shared_ptr< parameter_generator > generator, int param_count = 10 )
        {
            return unique_ptr< de_jung >( new de_jung( generator, param_count ) );
        }

        /**
         * @brief Changes the value of a given parameter. Derived classes must allways
         * call the base class method in order to maintain consistency.
         *
         * @param index the index of the parameter to be changed.
         * @param value the new parameter's value.
         */
        void set( int index, float value ) override
        {
            _params[ index ] = value;

            solution::set( index, value );
        }

        /**
         * @brief Returns the value of a given parameter.
         *
         * @param index the index of the parameter to be retorned.
         *
         * @return the current value of the parameter.
         */
        float get( int index ) const override
        {
            return _params[ index];
        }

        /**
         * @brief Returns the number of parameters of this solution.
         *
         * @return the number of parameters.
         */
        int size() const override
        {
            return _params.size();
        }

        /**
         * @brief Returns a reference to a vector containing all parameters of the solution.
         * If parameters are modified via this vector the solution should be notified called @see set_dirty()
         *
         * @return a reference to a vector containing all parameters.
         */
        vector<float>& get_parameters( ) override
        {
            return _params;
        }

        /**
         * @brief Creates an exact replica of this solution.
         *
         * @return a solution instance object that is equal to this solution.
         */
        virtual de_jung* clone() override
        {
            de_jung* result = new de_jung( _generator, _params.size() );

            result->_params = _params;

            return result;
        }

        /**
         * @brief Determines if the given object instance is assignable to this solution. A solution if assignable to
         * this one if is an de_jung solution and have the same number of parameters.
         *
         * @param a solution to check if it is assignable to this solution.
         *
         * @return true if the given solution is the given solution is assignable, false otherwise.
         */
        virtual bool assignable( solution &s ) const override
        {
            return ( typeid( s ) == typeid( de_jung ) ) &&
                   ( size()      == s.size() );
        }

    protected:

        de_jung( shared_ptr< parameter_generator > generator, int param_count  ) : solution( generator ), _params( param_count )
        {

        }

        /**
         * @brief Calculates the fitness of this solution which in this case is determined by the De'Jung function.
         * More information about De'Jung function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
         *
         * @return the fitness of this solution.
         */
        float calculate_fitness() override
        {
            float result = 0;

            for( int i = 0; i < _params.size(); i++ )
            {
                result += pow( _params[ i ], 2 );
            }

            return result;
        }

        /** This is a vector containing all the parameters of this solution. */
        vector< float >  _params;
    };
}
}
}

#endif
