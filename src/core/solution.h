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

#ifndef DNN_OPT_CORE_SOLUTION_H
#define DNN_OPT_CORE_SOLUTION_H

#include <stdexcept>
#include <random>

#include <src/core/layer.h>
#include <src/core/parameter_generator.h>

using namespace std;

namespace dnn_opt
{
namespace core
{
    /**
     * @brief This class represents a basic solution of any optimization problem.
     * In population based optimizations, this class can be seen as the abstract base class
     * that represent an individual in such population.
     * This class provides basic abstract methods and basic functionalities that derived classes
     * must implement. The most important feature of this class is that provides two virtual
     * methods that calculates the fitness or quality of this solution. Derived classes must provide
     * implementation for this methods according to the nature of the solution.
     *
     * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @date June, 2016
     * @version 1.0
     */
    class solution
    {

    public:

        /**
         * @brief Calculates the fitness of this solution. This function returns a precalculated fitness
         * if there have not been changes in its parameters otherwise calls @see calculate_fitness(). To
         * force the calculation of the fitness value call @see set_dirty().
         *
         * @return the fitness value of this solution.
         */
        float fitness()
        {
            if ( _modified == true )
            {
                _fitness = calculate_fitness();
            }

            _modified = false;

            return _fitness;
        }

        /**
         * @brief Changes the value of a given parameter. Derived classes are encouraged
         * call the base class method in order to maintain consistency.
         *
         * @param index the index of the parameter to be changed.
         * @param value the new parameter's value.
         */
        virtual void set( int /* index */, float /* value */ )
        {
            _modified           = true;
        }

        /**
         * @brief Changes the value of all the parameters of this solution.
         *
         * @param value the new parameter's value.
         */
        void set( float value )
        {
            _modified           = true;

            for( int i = 0; i < size(); i++ )
            {
                set( i, value );
            }
        }

        /**
         * @brief Returns the value of a given parameter.
         *
         * @param index the index of the parameter to be returned.
         *
         * @return the current value of the parameter.
         */
        virtual float get( int index ) const = 0;

        /**
         * @brief Returns the number of parameters of this solution.
         *
         * @return the number of parameters.
         */
        virtual int size() const = 0;

        /**
         * @brief Initialize all the parameters of this solution to random values.
         * Values are generated form the parameter_generator associated to this solution.
         */
        void init()
        {
            _modified           = true;

            for( int i = 0; i < size(); i++ )
            {
                set( i, _generator->generate() );
            }
        }

        /**
         * @brief `get_parameters` returns a reference to a vector containing all parameters of the solution.
         * If parameters are modified via this vector the solution should be notified calling @see set_dirty()
         *
         * @return a reference to a vector containing all parameters.
         */
        virtual vector<float>& get_parameters( ) = 0;

        /**
         * @brief `set_dirty` changes the state of this solution to force a
         * re-calculation of the current fitness value instead of using a pre-calculated value. T
         * his should be used whenever solution parameters have been changed via `solution::get_parameters()`.
         */
        void set_dirty()
        {
            _modified = true;
        }

        /**
         * @brief Creates an exact replica of this solution. The procedure to create a replica of a
         * solution depends on the nature of the solution and must be defined in derived classes.
         *
         * @return a solution instance that is equal to this solution.
         */
        virtual solution* clone() = 0;

        /**
         * @brief Determines if the given object instance is assignable to this solution. This
         * decision depends on the nature of the solution and must be defined in derived classes.
         * Usually this comprobation must check if the given solution is of the same type, the same
         * number of parameters.
         *
         * @param a solution to check if it is assignable to this solution.
         *
         * @return true if the given solution is the given solution is assignable,
         *         false otherwise.
         */
        virtual bool assignable( solution &s ) const = 0;

        /**
         * @brief get_generator returns a shared pointer containing a reference to the parameter_generator
         * that is currently used to generate the paraters of this solution.
         *
         * @return shared_ptr to the parameter_generator of this solution.
         */
        shared_ptr< parameter_generator > get_generator()
        {
            return _generator;
        }

    protected:

        /**
         * @brief Default constructor of solution. It's protected so derived class should implement a
         * factory pattern.
         */
        solution( shared_ptr< parameter_generator > generator ) : _generator( generator )
        {
            _modified           = true;
        }

        /**
         * @brief Calculates the fitness of this solution. The nature of the fitness depends
         * on the nature of the solution and must be implemented in derived classes.
         *
         * @return the fitness of this solution.
         */
        virtual float calculate_fitness() = 0;

        bool                                _modified;
        float                               _fitness;
        shared_ptr< parameter_generator >   _generator;
    };
}
}

#endif
