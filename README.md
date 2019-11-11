
# Welcome to dnn_opt

Welcome to dnn_opt, which states for deep neural network optimization. This is a C++11 library for high dimensional optimization and specifically for deep neural network optimization. However, it can be used for any kind of optimization problem.

The library will support sequential, multicore and GPU implementations for all the features. The current support for each implementation is shown bellow. 

* ![core implementation progress](http://progressed.io/bar/100) core: sequential implementation
* ![optimized implementation progress](http://progressed.io/bar/89) copt: parallel CPU implementation
* ![cuda implementation progress](http://progressed.io/bar/90) cuda: parallel GPU implementation

Keep reading to find out what you can currently do.

# Resources

The following resources are available for you:

- Quick start guide [see](docs/quick_start_guide.md)
- Development guidelines [see](docs/development_guidelines.md)
- Code standards [see](docs/code_standards.md)

# Examples

We feel that the best way of introduce our library features is by showing some examples. Please take a look at the following subsections. Additionally, feel free to take a look at our examples folder.

## Optimize sphere function with PSO

The following example is about optimizing a sphere function by using PSO meta-heuristic. Please take a look at the line starting with `using namespace dnn_opt::core`. There we are specifing to use a sequential implementation. If you have an NVIDIA GPU you can use it by changing the namespace to `using namespace dnn_opt::cuda`. If you have a multi-core CPU then use `using namespace dnn_opt::copt` and that is all.

````c++

#include <iostream>
#include <dnn_opt.h>

using namespace std;
using namespace dnn_opt::core;

int main(int argc, char** argv)
{
  /* generator that defines the search space */
  auto* generator = generators::uniform::make(-10.0f, 10.0f);

  /* set that contains the individuals of the population */
  auto* solutions = set<>::make(40);

  for(int i = 0; i < 40; i++)
  {
    solutions->add(solutions::de_jung::make(generator, 256));
  }

  /* random generation of initial population according the generator */
  solutions->generate();

  /* creating algorithm */
  auto* algorithm = algorithms::pso::make(solutions);

  /* optimize for 1000 iterations */
  algorithm->optimize_eval(1000, []()
  {
    /* this lamda is triggered at every iteration */
    /* return false any time to stop optimization */

    return true;
  });

  /* collect statics */
  cout << "Fitness: " << algorithm->get_best()->fitness() << endl;

  /* free memory */

  delete solutions->clean();
  delete algorithm;
  delete generator;

  return 0;
}

````

# Basic features

Take a look at the overall features provided by dnn_opt.

## Optimization algorithms

dnn_opt include some optimization algorithms that derive from the class `::algorithm` and are under the namespace `::algorithms::`. For the moment you can use the following optimization algorithms:

1. `::algorithms::firefly` which is the Firefly Algorithm
2. `::algorithms::pso` which is the Particle Swarm Optimization.
3. `::algorithms::cuckoo` which is the Cuckoo Search.

You can also find intresting how easely a new population-based meta-heuristic algorithm can be included. I'll create a doc section for this in the future.

## Benchmark function to test optimization algorithms

dnn_opt include some functions to run tests and do some benchmarking in high dimensional optimization. All the test functions are derived from a base class called `::solution` and reside under the namespace `::solutions::`. You are free to use:

1. `::solutions::ackley` which is the Ackley function.
2. `::solutions::de_jung` which is the De Jung function.
3. `::solutions::rastrigin` which is the Rastrigin function.
4. `::solutions::griewangk` which is the Griewangk function.
5. `::solutions::michaleicks` which is the Michalewicks function.
6. `::solutions::rosenbrock` which is the Rosenbrock function.
7. `::solutions::schwefel` which is the Schwefel function.
8. `::solutions::styblinski_tang` which is the Styblinski-Tang function.

Also, is very easy to include more benchmark functions. I'll provide a specific doc section for this too. As you may expect, there are also a special solution that stands for a dnn model which is: `::solutions::network`.

## Generators

Many population-based optimization methods require to randomly initialize its population. dnn_opt provides a mechanism to determine how to randomly generate the parameters of the solutions. This define the search space. All the parameter generators derive form the base class `::generator` and reside under the namespace `::generators::`. You are free to use the followings:

1. `::generators::normal` generate numbers with normal distribution in a given interval
2. `::generators::uniform` generate numbers with uniform distribution in a given interval
3. `::generators::group` generate numbers from a sequence of generators, you can use it to generate different dimensions of a solution vector with different distributions.

Extending new generators is straight-forward. I'll include more documentation about this but until that, you can refer to the documentation in the code.

## The network class

This is a special solution that models the neural network optimization surface. A neural network can be created in several ways and using parameters as stacked layers of procesing units and dnn_opt provides a way to accomplish this. Lets see what features you may find intresting.

### Activation functions

dnn_opt provides several activation functions that can be used by the processing units in a network. All activation functions derive from a base class called `::activation_function` and reside under the namespace `::activation_functions::`. You are free to use:

1. `::activation_functions::elu`
2. `::activation_functions::identity`
3. `::activation_functions::relu`
4. `::activation_functions::sigmoid`
5. `::activation_functions::tan_h`
6. `::activation_functions::softmax`

### Layers

Common dnn applications make use of several types of layers. dnn_opt provides several of them. All layers derive from a base class called `::layer` and reside under the namespace `::layers::`. You can use:

1. `::layers::fc`

Extending new layers is straight-forward. I'll include more documentation about the currently implemented layers and how to create new layers in the future. If you have any question please refer to the documentation in the code, everything is in there.

### Error functions

There are several ways to measure the error of a dnn. dnn_opt provides several error functions to accomplish this. All error functions derive from the base class `::error` and reside under the namespace `::errors::`. For the moment, you can use the followings:

1. `::errors::mse` for regression problems.
2. `::errors::overall_error` for classification problems.

Extending new error functions is straight-forward. I'll include more documentation about this but until that, you can refer to the documentation in the code.

### Input data

Training a DNN requires training patterns. dnn_opt load training patterns via classes that derives from `::reader` and recide under the namespace `::readers::`. For the moment we only provide a single class to do this job:

1. `::readers::file_reader` that reads from CSV files, please read the class documentation.

Extending new readers is straight-forward. I'll include more documentation about this but until that, you can refer to the documentation in the code.

# Related research papers

- Rojas-Delgado J., Milián Núñez V., Trujillo-Rasúa R., Bello R. (2019) Continuous Hyper-parameter Configuration for Particle Swarm Optimization via Auto-tuning. In: Lecture Notes in Computer Science, vol 11896. Springer, Cham. URL:  https://link.springer.com/chapter/10.1007/978-3-030-33904-3_43

- Rojas-Delgado J., Trujillo-Rasúa R., Bello R., Moya G.E.J. (2019) Video Popularity Forecasting to Improve Cache Miss Rate in Content Delivery Networks. In: Lecture Notes in Computer Science, vol 11896. Springer, Cham. URL: https://link.springer.com/chapter/10.1007/978-3-030-33904-3_73

- Rojas-Delgado J., Trujillo-Rasúa R. y Bello R. (2019) A continuation approach for training artificial neural networks with meta-heuristics. Pattern Recognition Letters, vol. 125, 373 - 380. Elseiver. URL: http://www.sciencedirect.com/science/article/pii/S0167865519301667

- Rojas-Delgado J., Trujillo-Rasúa R. (2018) Training Neural Networks by Continuation Particle Swarm Optimization. In: Lecture Notes in Computer Science, vol 11047. Springer, Cham. URL: https://link.springer.com/chapter/10.1007/978-3-030-01132-1_7

# How to contribute

Please take a look to the code standard and development guidelines in the documentation. You can take a look to the examples and read the documentation in the code. We'll be happy to hear from you!
