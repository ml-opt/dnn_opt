###############
# Getting Started
###############

## Installation

dnn_opt is currently available only from the sources. To install, clone the repo and fallow the next instructions.

````bash
git clone https://github.com/jairodelgado/dnn_opt
mkdir dnn_opt/bin
cd dnn_opt/bin
cmake ..
make
````

### Compilation options

Inside `CMakeLists.txt` file you can specify several options. By default only
the `dnn_opt::core` and examples layer are enabled. This is because you don't 
need to install any third-party software for it. For the rest of the layers you
will need to modify the following options:

````cmake
SET(ENABLE_COPT ON)
SET(ENABLE_CUDA ON)
````

### Installing third-party dependencies

For `dnn_opt::copt` and `dnn_opt::cuda` layers, you will need to install the
following dependencies in your system:

- The `dnn_opt::copt` layer depends on open source distributions of OpenMP and
BLAS. 
- The `dnn_opt::cuda` layer depends on open source distributions of Thrust, cuBLAS
and cuRand.

Make sure to install these dependencies in your system and that CMake is able
to find those in your path. 

## Folder structure and namespace resolution

The library comes with three main folders:

- `docs/` that contains documentation information
- `examples/` that contains out of the box and fully documented examples
- `src/` that contains the source code of the library

At the same time, the `src/` folder has the `src/dnn_opt.h` header file that
you need to include in your project. Then, a folder for each layer implementation
is presented.

Each layer have several packages that in general should be decoupled from one
another. An exception to this rule is the `base` package, that contains classes
that traverses all packages, for example: `solution` or `algorithm` classes.

The class `solution` is the base class for all solutions that are implemented in
the `solutions` package and the class `algorithm` is the base class for each
meta-heuristic algorithm implemented in the package `algorithms`. In general,
each package has its own namespace as well named in the same way. For example, to
use the sequential PSO algorithm you need to use the following namespace 
resolution: `dnn_opt::core::algorithms::pso`.

## Basic usage

Once you compile the library one or several libraries binaries are generated. In
your main file you need to include the header `dnn_opt.h` that at the same time
will include all necessary headers.

We recommend to solve the library namespace until the layers name, for example,
`using namespace dnn_opt::core`. Then use the library functionality at free
will.

````c++
#include <dnn_opt.h>

using namespace dnn_opt::core;

int main()
{
  auto* gen = generators::uniform::make(-1.0f, 1.0f);
  ...
}
````

That way, the code becomes less verbose and you won't make the mistake of
mixing classes from several layers. Although, it may be possible to do such
mix the library is not intended to do so. Hence, is highly discouraged.

Functionality is provided through the same clean and transparent API for every 
layer in the library. You do not need to know the details of the layer 
implementation, but the interface functionality and everything works out of 
the box. This means that if you know how to use the library for the `dnn_opt::core`
layer, then, you know everything!

### Basic concepts

The library has a small number of basic concepts. The **solutions** are classes to
hold a list of parameters and a fitness function to evaluate the quality of such
parameters. An **algorithm** is a procedure to change the solution/s parameters
in order to improve the fitness function of such solution/s.

A **network** is a special kind of solution that provides support for neural
networks in the library. Networks are assembled with stacked **layers** of
neurons. In addition, they have an **error** function to calculate loss and a 
**reader** that hold instances. The error and the reader are used to calculate 
the fitness function value for the network.

All the library functionalities are provided by extending this simple set
of base classes. For example, to support convolutional neural networks, we extend
the `layer` base class and to provide the mean squared error loss function we
extend the `error` base class.

### Simple optimization routine

The library has simple usage routine to perform optimization. That routine
can be described in the following steeps:

1. Create the generator for the solutions, the generator is used to create 
initial population randomly
2. Create the solution set that will hold the population of solutions
3. Populate the solution set with the solutions we want to optimize
4. Generate initial population
5. Create the optimization algorithm you want to use
6. Perform optimization
7. Destroy the memory you have allocated

## Examples

Please take a look at the examples folder. There you can find several non-trivial
examples of the library usage.

