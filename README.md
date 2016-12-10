[![Build Status](https://travis-ci.org/jairodelgado/dnn_opt.svg?branch=master)](https://travis-ci.org/jairodelgado/dnn_opt)

# Welcome to DNN_OPT

Welcome to dnn_opt, which states for deep neural network optimization. This is a C++11 header only library for high dimensional optimization and specifically for deep neural network optimization. High dimensional optimization is difficult to accomplish due to curse of dimensionality and high temporal and space complexity. Training a deep neural network is in fact an NP-hard optimization problem. 

For the moment, we are only providing basic library functionalities, but in the future dnn_opt will support multicore, GPU and distributed implementations for all the features, which for the moment only include sequential implementations. Keep reading to find out what you can currently do.

# Optimization algorithms

dnn_opt include some optimization algorithms that derive from the class `dnn_opt::core::algorithm` and are under the namespace `dnn_opt::core::algorithms::`. For the moment you can use the following optimization algorithms:

1. `dnn_opt::core::algorithms::firefly` which is the Firefly Algorithm
2. `dnn_opt::core::algorithms::pso` which is the Particle Swarm Optimization.
3. `dnn_opt::core::algorithms::cuckoo` which is the Cuckoo Search.
4. `dnn_opt::core::algorithms::gray_wolf` which is the Gray Wolf Optimization.

You can also find intresting how easely a new population-based meta-heuristic algorithm can be included. I'll create a doc section for this in the future.

# Benchmark function for test optimization algorithms

dnn_opt include some functions to run tests and do some benchmarking in high dimensional optimization. All the test functions are derived from a base class called `dnn_opt::core::solution` and reside under the namespace `dnn_opt::core::solutions::`. You are free to use:

1. `dnn_opt::core::solutions::ackley` which is the Ackley function.
2. `dnn_opt::core::solutions::de_jung` which is the De Jung function.
3. `dnn_opt::core::solutions::rastrigin` which is the Rastrigin function.
4. `dnn_opt::core::solutions::griewangk` which is the Griewangk function.
5. `dnn_opt::core::solutions::michaleicks` which is the Michalewicks function.
6. `dnn_opt::core::solutions::rosenbrock` which is the Rosenbrock function.
7. `dnn_opt::core::solutions::schwefel` which is the Schwefel function.
8. `dnn_opt::core::solutions::styblinski_tang` which is the Styblinski-Tang function.

Also, is very easy to include more benchmark functions. I'll provide a specific doc section for this too. As you may expect, there are also a special solution that stands for a dnn model which is: `dnn_opt::core::solutions::network`.

# The `tiny_dnn::core::solutions::network`

This is a special solution that models the dnn optimization surface. A dnn can be created in several ways and using parameters as stacked layers of procesing units and dnn_opt provides a way to accomplish this. Lets see what features you may find intresting.

## Activation functions.

dnn_opt provides several activation functions that can be used by the processing units in a network. All activation functions derive from a base class called `dnn_opt::core::activation_function` and reside under the namespace `dnn_opt::core::activation_functions::`. You are free to use:

1. `dnn_opt::core::activation_functions::elu`
2. `dnn_opt::core::activation_functions::identity`
3. `dnn_opt::core::activation_functions::relu`
4. `dnn_opt::core::activation_functions::sigmoid`
5. `dnn_opt::core::activation_functions::tan_h`
6. `dnn_opt::core::activation_functions::softmax`

## Layers

Common dnn applications make use of several types of layers. dnn_opt provides several of them. All layers derive from a base class called `dnn_opt::core::layer` and reside under the namespace `dnn_opt::core::layers::`. You can use:

1. `dnn_opt::core::layers::fully_connected`
2. `dnn_opt::core::layers::convolutional`
3. `dnn_opt::core::layers::average pooling`
4. `dnn_opt::core::layers::max_pooling`
5. `dnn_opt::core::layers::discretization`

Extending new layers is straight-forward. I'll include more documentation about the currently implemented layers and how to create new layers in the future. If you have any question please refer to the documentation in the code, everything is in there.

## Parameter generators

Many population-based optimization methods require to randomly initialize its population. dnn_opt provides a mechanism for determine how to randomly generate the parameters of the solutions. All the parameter generators derive form the base class `dnn_opt::core::parameter_generator` and reside under the namespace `dnn_opt::core::parameter_generators::`. You are free to use the followings:

1. `dnn_opt::core::parameter_generators::normal`
2. `dnn_opt::core::parameter_generators::uniform`

Extending new paramter generators is straight-forward too. I'll include more documentation about this but until that, you can refer to the documentation in the code.

## Error functions

There are several ways to measure the error of a dnn. dnn_opt provides several error functions to accomplish this. All error functions derive from the base class `dnn_opt::core::error_function` and reside under the namespace `dnn_opt::core::activation_functions::`. For the moment, you can use the followings:

1. `dnn_opt::core::activation_functions::mse` for regression problems.
2. `dnn_opt::core::activation_functions::overall_error` for classification problems.

Extending new error functions is straight-forward. I'll include more documentation about this but until that, you can refer to the documentation in the code.

# Input readers

Training a dnn requires training patterns. dnn_opt load training patterns via classes that derives from `dnn_opt::core::reader` and recide under the namespace `dnn_opt::core::io::`. For the moment we only provide a single class to do this job:

1. `dnn_opt::core::io::file_reader`

Extending new readers is straight-forward. I'll include more documentation about this but until that, you can refer to the documentation in the code.

# Samplers

This is an experimental feature and it will be documented in the future. Please just refer to the examples and see how to use it.

# Examples

In this section you can see a basic example of how to use dnn_opt.

## Multi-layer perceptron with Firefly Algorithm

In this example we will train a multi-layer perceptron with the firefly algorithm. Please make sure to provide the library with the correct format for the input file and a regresion problem. Also, tune the hyper-parameters of the Firefly Algorithm correctly.  See the code documentation for the `dnn_opt::core::io::file_reader` class to find out the correct input file format. You can use some of the following prepared files that are modified from the [UCI repository for machine learning](http://archive.ics.uci.edu/ml/datasets.html):

1. [Wine Quality Dataset](docs/regression_problems/winequality-white.csv).
2. [Concrete Compressive Strenght](docs/regression_problems/concrete.csv). 
3. [Combined Cycle Power Plant](docs/regression_problems/ccpp.csv).

````c++
#include <iostream>
#include <memory>

#include "src/io/file_reader.h"
#include "src/core/sampler.h"
#include "src/core/solution_set.h"
#include "src/metaheuristic/firefly.h"
#include "src/solution/network.h"
#include "src/model/layer/fully_connected.h"
#include "src/model/error_function/mse.h"
#include "src/model/activation_function/tan_h.h"
#include "src/model/parameter_generator/uniform.h"

using namespace dnn_opt::core;

int main()
{
  auto reader     = io::file_reader::make( "./winequality-white.csv" );
  auto generator  = parameter_generators::normal::make( 0, 1 );
  auto sampler    = sampler::make( 0, reader->get_input_data(), reader->get_output_data() );
  auto error      = error_functions::mse::make();
  auto solutions  = solution_set::make(10);

  for( int i = 0; i < 10; i++ )
  {
    auto net      = solutions::network::make( true, sampler, error, generator );

    ( *net ) << layers::fully_connected::make( 11, 10, activation_functions::tan_h::make() )
             << layers::fully_connected::make( 10, 10, activation_functions::tan_h::make() )
             << layers::fully_connected::make( 10, 1, activation_functions::tan_h::make() );

    solutions->add( std::move( net ) ;
  }

  solutions->init();

  auto algorithm  = algorithms::firefly::make(0.5, 0.75, 0.35, std::move( solutions ) );

  for( int i = 0; i < 50; i++ )
  {
      algorithm->optimize( 10 );
  }

  /* This is the average MSE error of the entire population */
  std::cout << algorithm->get_solutions()->fitness();
}

````

# How to contribute

For the moment I won't be accepting contributions. First I'll complete some documentation, prepare some design guidelines and code standards. Any way, if you still want to contribute here is my [blog](https://jairodelgado.github.io) and the contact information is there. Thank you very much for reading. Enjoy!