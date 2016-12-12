# namespace `ann_pt::core::layers` {#namespaceann__pt_1_1core_1_1layers}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`ann_pt::core::layers::average_pooling`](#classann__pt_1_1core_1_1layers_1_1average__pooling)    | 
# class `ann_pt::core::layers::average_pooling` {#classann__pt_1_1core_1_1layers_1_1average__pooling}

```
class ann_pt::core::layers::average_pooling
  : public dnn_opt::core::layer
```  



: Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

October, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline vector< float > propagate(const vector< float > & input,const vector< float > & params,unsigned int start,unsigned int end)` | 
`public inline virtual unsigned int count() const` | count returns the number of parameters that is required by this layer. This is [input_dimension()](#classdnn__opt_1_1core_1_1layer_1ab502eb1b011bdf8008e6301665d789a0) * [output_dimension()](#classdnn__opt_1_1core_1_1layer_1a64063d73e577923444900256512d364c) since this is a fully connected layer.
`protected inline  average_pooling(unsigned int in_height,unsigned int in_width,unsigned int in_depth,unsigned int w_height,unsigned int w_width,unsigned int stride)` | 

## Members

#### `public inline vector< float > propagate(const vector< float > & input,const vector< float > & params,unsigned int start,unsigned int end)` {#classann__pt_1_1core_1_1layers_1_1average__pooling_1aca1cc6c2f55c58dfc895f67d317b0d74}





#### `public inline virtual unsigned int count() const` {#classann__pt_1_1core_1_1layers_1_1average__pooling_1a3da6db1171eb78bb0ae310e6fc12a786}

count returns the number of parameters that is required by this layer. This is [input_dimension()](#classdnn__opt_1_1core_1_1layer_1ab502eb1b011bdf8008e6301665d789a0) * [output_dimension()](#classdnn__opt_1_1core_1_1layer_1a64063d73e577923444900256512d364c) since this is a fully connected layer.

#### Returns
the number of parameters.

#### `protected inline  average_pooling(unsigned int in_height,unsigned int in_width,unsigned int in_depth,unsigned int w_height,unsigned int w_width,unsigned int stride)` {#classann__pt_1_1core_1_1layers_1_1average__pooling_1ac24f5206a26cfd24a57c455b951c39d9}





# namespace `dnn_opt::core` {#namespacednn__opt_1_1core}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`dnn_opt::core::activation_function`](#classdnn__opt_1_1core_1_1activation__function)    | The [activation_function](#classdnn__opt_1_1core_1_1activation__function) class is intended to provide an interface for custom activation functions that can be used by an artificial neural network. Derived classes shuld expect to receive weighted summatory outputs of units in a layer. In case of layers outputs that have some specific spatial arrangement, like convolutional layers, specialiced activation functions should consider such implementation details about its output.
`class `[`dnn_opt::core::algorithm`](#classdnn__opt_1_1core_1_1algorithm)    | 
`class `[`dnn_opt::core::error_function`](#classdnn__opt_1_1core_1_1error__function)    | The [error_function](#classdnn__opt_1_1core_1_1error__function) class is a virtual class intended as a superclass for all error functions that can be implemented for an artificial neural network.
`class `[`dnn_opt::core::layer`](#classdnn__opt_1_1core_1_1layer)    | The layer class is a superclass for all types of layers that could be implemented for an artificial neural network. Deep learning is a stack of layers in cascade that transform an input signal into a high level representation of such input signal.
`class `[`dnn_opt::core::parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)    | The [parameter_generator](#classdnn__opt_1_1core_1_1parameter__generator) class is intended as superclass to implement custom parameters generators. A [parameter_generator](#classdnn__opt_1_1core_1_1parameter__generator) basically generates random numbers to be used as parameters of an optimization solution.
`class `[`dnn_opt::core::reader`](#classdnn__opt_1_1core_1_1reader)    | This class is intended to provide an interface for reading and loading patterns into the library. A pattern is a `vector< float >`.
`class `[`dnn_opt::core::sampler`](#classdnn__opt_1_1core_1_1sampler)    | The sampler class is intended to maintain samples of training data for neural networks that is going to be used while optimization. Basically contains getters methods of input and output training patterns and getters of input and output training patterns of a random subset of the original data. A training pattern is a `vector< float >.`
`class `[`dnn_opt::core::solution`](#classdnn__opt_1_1core_1_1solution)    | This class represents a basic solution of any optimization problem. In population based optimizations, this class can be seen as the abstract base class that represent an individual in such population. This class provides basic abstract methods and basic functionalities that derived classes must implement. The most important feature of this class is that provides two virtual methods that calculates the fitness or quality of this solution. Derived classes must provide implementation for this methods according to the nature of the solution.
`class `[`dnn_opt::core::solution_set`](#classdnn__opt_1_1core_1_1solution__set)    | The [solution_set](#classdnn__opt_1_1core_1_1solution__set) class is intended to manage a set of optimization solutions for a determined optimization problem. This a helper class that can be usefull for population based optimization metaheuristics.
# class `dnn_opt::core::activation_function` {#classdnn__opt_1_1core_1_1activation__function}


The [activation_function](#classdnn__opt_1_1core_1_1activation__function) class is intended to provide an interface for custom activation functions that can be used by an artificial neural network. Derived classes shuld expect to receive weighted summatory outputs of units in a layer. In case of layers outputs that have some specific spatial arrangement, like convolutional layers, specialiced activation functions should consider such implementation details about its output.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

September, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public float activation(const vector< float > & output,int index)` | activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

## Members

#### `public float activation(const vector< float > & output,int index)` {#classdnn__opt_1_1core_1_1activation__function_1a567055c96430cac94f49a4d499a79388}

activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

#### Parameters
* `output` `a` vector containing a layer output in terms of weighted summatory. 


* `index` `the` position of the corresponding unit in the layer which output is going to be calculated.





#### Returns
the activation value of the processing unit.

# class `dnn_opt::core::algorithm` {#classdnn__opt_1_1core_1_1algorithm}




This class represents an abstract optimization algorithm capable of define the basic functionalities of any meta-heuristic. In order to extend the library, new algorithms shuold derive from this class.

Derived classes are free to introduce custom algorithm parameters on its constructor but always have to provide a [solution_set](#classdnn__opt_1_1core_1_1solution__set) class in order to define the individuals to be optimized. In case that the new algorithm is not population based, it can be used a [solution_set](#classdnn__opt_1_1core_1_1solution__set) class with only one individual.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

June, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public void optimize()` | Abstract method that implements a single steep of optimization.
`public inline void optimize(int count)` | This method performs multiple steeps of optimization of this optimization algorithm.
`public inline unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > & get_solutions()` | Returns the [solution_set](#classdnn__opt_1_1core_1_1solution__set) used by this algorithm.
`protected unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > _solutions` | 
`protected inline  algorithm(unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > solutions)` | The basic contructor for an optimization algorithm. In case that the new algorithm is not population based we strongly recommend to use a [solution_set](#classdnn__opt_1_1core_1_1solution__set) class with only one individual.

## Members

#### `public void optimize()` {#classdnn__opt_1_1core_1_1algorithm_1af30d7a6dddcb5cc443468adbfbaeafdc}

Abstract method that implements a single steep of optimization.



#### `public inline void optimize(int count)` {#classdnn__opt_1_1core_1_1algorithm_1aac96441f019433d0f19a487e895db424}

This method performs multiple steeps of optimization of this optimization algorithm.

#### Parameters
* `count` is the number of steeps to perform.

#### `public inline unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > & get_solutions()` {#classdnn__opt_1_1core_1_1algorithm_1a077a1b25fecc964007bc3ac509661531}

Returns the [solution_set](#classdnn__opt_1_1core_1_1solution__set) used by this algorithm.

#### Returns
an unique_ptr pointing to an instance of the solution set used by this algorithm.

#### `protected unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > _solutions` {#classdnn__opt_1_1core_1_1algorithm_1aee3e33b9184023b7d5011fea0eac3ad0}





#### `protected inline  algorithm(unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > solutions)` {#classdnn__opt_1_1core_1_1algorithm_1ac7f876b89ff42ad88bfd811112984749}

The basic contructor for an optimization algorithm. In case that the new algorithm is not population based we strongly recommend to use a [solution_set](#classdnn__opt_1_1core_1_1solution__set) class with only one individual.

#### Parameters
* `solutions` is the set of individuals to optimize.

# class `dnn_opt::core::error_function` {#classdnn__opt_1_1core_1_1error__function}


The [error_function](#classdnn__opt_1_1core_1_1error__function) class is a virtual class intended as a superclass for all error functions that can be implemented for an artificial neural network.

Jairo Rojas-Delgado 

1.0 

June, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual float error(const vector< vector< float > > & real,const vector< vector< float > > & expected)` | Calculates the error value between the real output of an artificial neural network and an expected output.

## Members

#### `public inline virtual float error(const vector< vector< float > > & real,const vector< vector< float > > & expected)` {#classdnn__opt_1_1core_1_1error__function_1a144a6b2bbc17d90d687ae43013323414}

Calculates the error value between the real output of an artificial neural network and an expected output.

#### Parameters
* `real` a multi-target output of an artificial neural network resulting of the propagation of several training / validation patterns.


* `expected` a multi-target expected output of an artificial neural network.





#### Returns
the error value between the real output of the network and the expected output.


#### Parameters
* `assertion` if `real.size() != expected.size()`

# class `dnn_opt::core::layer` {#classdnn__opt_1_1core_1_1layer}


The layer class is a superclass for all types of layers that could be implemented for an artificial neural network. Deep learning is a stack of layers in cascade that transform an input signal into a high level representation of such input signal.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

1.0 

June, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void propagate(const vector< float > & input,const vector< float > & params,int start,int end)` | propagate is an interface to propagate a given input signal to an output. The output value is managed by this class, hence to obtain the current output value resulting of calling this method you should call
`public inline int input_dimension() const` | input_dimension specifies the number of values this layer accepts as input.
`public inline int output_dimension() const` | output_dimension specifies the number of values this layer produces as output.
`public int count() const` | count returns the number of parameters that is required by this layer.
`public const vector< float > & output() const` | output returns a vector vith the output values of this layer after
`public `[`layer`](#classdnn__opt_1_1core_1_1layer)` * clone()` | clone returns an exact copy of this layer.
`protected int _in_dim` | 
`protected int _out_dim` | 
`protected shared_ptr< `[`activation_function`](#classdnn__opt_1_1core_1_1activation__function)` > _AF` | 
`protected inline  layer(int in_dim,int out_dim,shared_ptr< `[`activation_function`](#classdnn__opt_1_1core_1_1activation__function)` > AF)` | layer is the basic contructor for this class. Is intended to be used by derived classes that implements the factory pattern.

## Members

#### `public inline virtual void propagate(const vector< float > & input,const vector< float > & params,int start,int end)` {#classdnn__opt_1_1core_1_1layer_1a95b78ac43f02316b9ad848070f8e5708}

propagate is an interface to propagate a given input signal to an output. The output value is managed by this class, hence to obtain the current output value resulting of calling this method you should call

**See also**: [output()](#classdnn__opt_1_1core_1_1layer_1a840b60d0d17291c4bdd94e53412840db). Derived classes are encouraged to call this method for validation.

**Warning:** the number of values in the input vector should ve equal to 
**See also**: [input_dimension()](#classdnn__opt_1_1core_1_1layer_1ab502eb1b011bdf8008e6301665d789a0). At the same time start and end must be valid indexes in the params vector, start <= end. Failing these requirements produces assertion error.


#### Parameters
* `input` is the input signal to be propagated. 


* `params` is a vector containing a parameter list to be used for this layer. 


* `start` is where the parameters of this layer starts in the params vector. 


* `end` is where the paramters of this layer ends in the params vector.





#### Parameters
* `assertion` if any of the previously mentioned conditions fails.

#### `public inline int input_dimension() const` {#classdnn__opt_1_1core_1_1layer_1ab502eb1b011bdf8008e6301665d789a0}

input_dimension specifies the number of values this layer accepts as input.

#### Returns
the number of input dimensions.

#### `public inline int output_dimension() const` {#classdnn__opt_1_1core_1_1layer_1a64063d73e577923444900256512d364c}

output_dimension specifies the number of values this layer produces as output.

#### Returns
the number of output dimensions.

#### `public int count() const` {#classdnn__opt_1_1core_1_1layer_1a96ffec579c43ae780dc5010367007243}

count returns the number of parameters that is required by this layer.

#### Returns
the number of parameters.

#### `public const vector< float > & output() const` {#classdnn__opt_1_1core_1_1layer_1a840b60d0d17291c4bdd94e53412840db}

output returns a vector vith the output values of this layer after

**See also**: [propagate()](#classdnn__opt_1_1core_1_1layer_1a95b78ac43f02316b9ad848070f8e5708) was called. Calling this method before 


**See also**: [propagate()](#classdnn__opt_1_1core_1_1layer_1a95b78ac43f02316b9ad848070f8e5708) produces inconsistent results.


#### Returns
a vector with 


**See also**: [output_dimension()](#classdnn__opt_1_1core_1_1layer_1a64063d73e577923444900256512d364c) values corresponging to the [output](#classdnn__opt_1_1core_1_1layer_1a840b60d0d17291c4bdd94e53412840db) of this [layer](#classdnn__opt_1_1core_1_1layer).

#### `public `[`layer`](#classdnn__opt_1_1core_1_1layer)` * clone()` {#classdnn__opt_1_1core_1_1layer_1a114bd27dbeee469e4aed8596cbfe845c}

clone returns an exact copy of this layer.

#### Returns
a pointer to the copyy of this layer.

#### `protected int _in_dim` {#classdnn__opt_1_1core_1_1layer_1abe02e65c6d798b972ae218a38dea9c2a}





#### `protected int _out_dim` {#classdnn__opt_1_1core_1_1layer_1a5f196d974a29123ab153e06474e59271}





#### `protected shared_ptr< `[`activation_function`](#classdnn__opt_1_1core_1_1activation__function)` > _AF` {#classdnn__opt_1_1core_1_1layer_1a448378ffe29577f79bfa5d2876b7aa13}





#### `protected inline  layer(int in_dim,int out_dim,shared_ptr< `[`activation_function`](#classdnn__opt_1_1core_1_1activation__function)` > AF)` {#classdnn__opt_1_1core_1_1layer_1a1aecdb44c56490376c9ad9139e443b4f}

layer is the basic contructor for this class. Is intended to be used by derived classes that implements the factory pattern.

#### Parameters
* `in_dim` the number of values that this layers expects as input. 


* `out_dim` the number of values that this layer will produce as output. 


* `AF` the activation function that will be used by the units of this layer.

# class `dnn_opt::core::parameter_generator` {#classdnn__opt_1_1core_1_1parameter__generator}


The [parameter_generator](#classdnn__opt_1_1core_1_1parameter__generator) class is intended as superclass to implement custom parameters generators. A [parameter_generator](#classdnn__opt_1_1core_1_1parameter__generator) basically generates random numbers to be used as parameters of an optimization solution.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

1.0 

June, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public float generate()` | generate a single parameter.

## Members

#### `public float generate()` {#classdnn__opt_1_1core_1_1parameter__generator_1acac700e507f72b31e4d9bf905df48cd7}

generate a single parameter.

#### Returns
the value of the parameter.

# class `dnn_opt::core::reader` {#classdnn__opt_1_1core_1_1reader}


This class is intended to provide an interface for reading and loading patterns into the library. A pattern is a `vector< float >`.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

1.0 

June, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public vector< vector< float > > & get_input_data()` | Returns a vector containing input patterns.
`public vector< vector< float > > & get_output_data()` | Returns a vector containing output patterns.

## Members

#### `public vector< vector< float > > & get_input_data()` {#classdnn__opt_1_1core_1_1reader_1ae246f12c6a42a82aa938175d1e240258}

Returns a vector containing input patterns.

#### Returns
a vector of input patterns.

#### `public vector< vector< float > > & get_output_data()` {#classdnn__opt_1_1core_1_1reader_1a099168c7b3db71a76ec8aa4c3a246861}

Returns a vector containing output patterns.

#### Returns
a vector of output patterns.

# class `dnn_opt::core::sampler` {#classdnn__opt_1_1core_1_1sampler}


The sampler class is intended to maintain samples of training data for neural networks that is going to be used while optimization. Basically contains getters methods of input and output training patterns and getters of input and output training patterns of a random subset of the original data. A training pattern is a `vector< float >.`

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

June, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline const vector< vector< float > > & input() const` | Returns the original set of input training patterns.
`public inline const vector< vector< float > > & output() const` | Returns the original set of output training patterns.
`public inline const vector< vector< float > > & sample_input()` | Returns a random subset of input training patterns.
`public inline const vector< vector< float > > & sample_output() const` | Returns the random subset of output training patterns.
`public inline int size() const` | Returns the number of training patterns in the original dataset.
`public inline int sample_size() const` | Returns the number of training patterns in the random subset.
`public inline void sample()` | sample forces the re-generation of the random sample.
`protected vector< vector< float > > _input` | 
`protected vector< vector< float > > _output` | 
`protected vector< vector< float > > _sample_input` | 
`protected vector< vector< float > > _sample_output` | 
`protected int _size` | 
`protected inline  sampler(int size,const vector< vector< float > > & input,const vector< vector< float > > & output)` | Creates a sampler object with a random subset with the given number of training patterns from the original training data.

## Members

#### `public inline const vector< vector< float > > & input() const` {#classdnn__opt_1_1core_1_1sampler_1a67ff18a032269481525ec94e008898b2}

Returns the original set of input training patterns.

#### Returns
a vector with the input training patterns.

#### `public inline const vector< vector< float > > & output() const` {#classdnn__opt_1_1core_1_1sampler_1a17eb0f8ac7ba5fabe6cc9adf29b07464}

Returns the original set of output training patterns.

#### Returns
a vector with the output training patterns.

#### `public inline const vector< vector< float > > & sample_input()` {#classdnn__opt_1_1core_1_1sampler_1acc32dd5a648fb954a492180e26be0884}

Returns a random subset of input training patterns.

#### Returns
a vector with the input training patterns.

#### `public inline const vector< vector< float > > & sample_output() const` {#classdnn__opt_1_1core_1_1sampler_1a89234c744f6b78d1b5fab94a73998aaf}

Returns the random subset of output training patterns.

#### Returns
a vector with the output training patterns.

#### `public inline int size() const` {#classdnn__opt_1_1core_1_1sampler_1a1af41c9c361490109a7b9a8ba9440b47}

Returns the number of training patterns in the original dataset.

#### Returns
the number of training patterns.

#### `public inline int sample_size() const` {#classdnn__opt_1_1core_1_1sampler_1adde8bd8da03dfe28485fc7f008c53a5b}

Returns the number of training patterns in the random subset.

#### Returns
the number of training patterns.

#### `public inline void sample()` {#classdnn__opt_1_1core_1_1sampler_1ac63a23247f94e82219c9e2ef56dc0202}

sample forces the re-generation of the random sample.



#### `protected vector< vector< float > > _input` {#classdnn__opt_1_1core_1_1sampler_1a1c33bd8b435cf78d5a78c56b4c109d2d}





#### `protected vector< vector< float > > _output` {#classdnn__opt_1_1core_1_1sampler_1ace71e9252a3181e1c9d207a766890e91}





#### `protected vector< vector< float > > _sample_input` {#classdnn__opt_1_1core_1_1sampler_1af0f5f24f83db1668d033417c20312e3b}





#### `protected vector< vector< float > > _sample_output` {#classdnn__opt_1_1core_1_1sampler_1ab04f5fb829648770e74fb8a60fc933b8}





#### `protected int _size` {#classdnn__opt_1_1core_1_1sampler_1a11861d448a1d8dff2b9abf2d66af710b}





#### `protected inline  sampler(int size,const vector< vector< float > > & input,const vector< vector< float > > & output)` {#classdnn__opt_1_1core_1_1sampler_1ac3d8a2e88ae6790dd02a56ce8459d9f8}

Creates a sampler object with a random subset with the given number of training patterns from the original training data.

#### Parameters
* `size` the number of training patterns to include in the random subset. 


* `input` the full set of input training patterns. 


* `output` the full set of output training patterns.





#### Parameters
* `assertion` if input and output does not have the same size or if the size of the subset is bigger than the number of training patterns.

# class `dnn_opt::core::solution` {#classdnn__opt_1_1core_1_1solution}


This class represents a basic solution of any optimization problem. In population based optimizations, this class can be seen as the abstract base class that represent an individual in such population. This class provides basic abstract methods and basic functionalities that derived classes must implement. The most important feature of this class is that provides two virtual methods that calculates the fitness or quality of this solution. Derived classes must provide implementation for this methods according to the nature of the solution.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

June, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline float fitness()` | Calculates the fitness of this solution. This function returns a precalculated fitness if there have not been changes in its parameters otherwise calls.
`public inline virtual void set(int,float)` | Changes the value of a given parameter. Derived classes are encouraged call the base class method in order to maintain consistency.
`public inline void set(float value)` | Changes the value of all the parameters of this solution.
`public float get(int index) const` | Returns the value of a given parameter.
`public int size() const` | Returns the number of parameters of this solution.
`public inline void init()` | Initialize all the parameters of this solution to random values. Values are generated form the [parameter_generator](#classdnn__opt_1_1core_1_1parameter__generator) associated to this solution.
`public vector< float > & get_parameters()` | `get_parameters` returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified calling
`public inline void set_dirty()` | `set_dirty` changes the state of this solution to force a re-calculation of the current fitness value instead of using a pre-calculated value. T his should be used whenever solution parameters have been changed via `[solution::get_parameters()](#classdnn__opt_1_1core_1_1solution_1a288dca51ef1f9f6baeb9291bf962203f)`.
`public `[`solution`](#classdnn__opt_1_1core_1_1solution)` * clone()` | Creates an exact replica of this solution. The procedure to create a replica of a solution depends on the nature of the solution and must be defined in derived classes.
`public bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` | Determines if the given object instance is assignable to this solution. This decision depends on the nature of the solution and must be defined in derived classes. Usually this comprobation must check if the given solution is of the same type, the same number of parameters.
`public inline shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > get_generator()` | get_generator returns a shared pointer containing a reference to the [parameter_generator](#classdnn__opt_1_1core_1_1parameter__generator) that is currently used to generate the paraters of this solution.
`protected bool _modified` | 
`protected float _fitness` | 
`protected shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > _generator` | 
`protected inline  solution(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator)` | Default constructor of solution. It's protected so derived class should implement a factory pattern.
`protected float calculate_fitness()` | Calculates the fitness of this solution. The nature of the fitness depends on the nature of the solution and must be implemented in derived classes.

## Members

#### `public inline float fitness()` {#classdnn__opt_1_1core_1_1solution_1a3b5f0256cd00556bb8bc65218e2f786d}

Calculates the fitness of this solution. This function returns a precalculated fitness if there have not been changes in its parameters otherwise calls.

**See also**: [calculate_fitness()](#classdnn__opt_1_1core_1_1solution_1af3dbd317aabae744d814c506577620fb). To force the calculation of the [fitness](#classdnn__opt_1_1core_1_1solution_1a3b5f0256cd00556bb8bc65218e2f786d) value call 


**See also**: [set_dirty()](#classdnn__opt_1_1core_1_1solution_1a44232576e3a8d1b616f747a24180f397).


#### Returns
the fitness value of this solution.

#### `public inline virtual void set(int,float)` {#classdnn__opt_1_1core_1_1solution_1afb1ba98848b4b7c4a0e2441e9fc6bba5}

Changes the value of a given parameter. Derived classes are encouraged call the base class method in order to maintain consistency.

#### Parameters
* `index` the index of the parameter to be changed. 


* `value` the new parameter's value.

#### `public inline void set(float value)` {#classdnn__opt_1_1core_1_1solution_1a2b8eee4e804f86d6f2da40563331a4e4}

Changes the value of all the parameters of this solution.

#### Parameters
* `value` the new parameter's value.

#### `public float get(int index) const` {#classdnn__opt_1_1core_1_1solution_1a11b711e554f5b3461943a04d0488abf1}

Returns the value of a given parameter.

#### Parameters
* `index` the index of the parameter to be returned.





#### Returns
the current value of the parameter.

#### `public int size() const` {#classdnn__opt_1_1core_1_1solution_1a6b4d9d7f7711de9ca4aa90e65ded5d9a}

Returns the number of parameters of this solution.

#### Returns
the number of parameters.

#### `public inline void init()` {#classdnn__opt_1_1core_1_1solution_1a769822b0c8bc5067c23eed15277b372e}

Initialize all the parameters of this solution to random values. Values are generated form the [parameter_generator](#classdnn__opt_1_1core_1_1parameter__generator) associated to this solution.



#### `public vector< float > & get_parameters()` {#classdnn__opt_1_1core_1_1solution_1a288dca51ef1f9f6baeb9291bf962203f}

`get_parameters` returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified calling

**See also**: [set_dirty()](#classdnn__opt_1_1core_1_1solution_1a44232576e3a8d1b616f747a24180f397)


#### Returns
a reference to a vector containing all parameters.

#### `public inline void set_dirty()` {#classdnn__opt_1_1core_1_1solution_1a44232576e3a8d1b616f747a24180f397}

`set_dirty` changes the state of this solution to force a re-calculation of the current fitness value instead of using a pre-calculated value. T his should be used whenever solution parameters have been changed via `[solution::get_parameters()](#classdnn__opt_1_1core_1_1solution_1a288dca51ef1f9f6baeb9291bf962203f)`.



#### `public `[`solution`](#classdnn__opt_1_1core_1_1solution)` * clone()` {#classdnn__opt_1_1core_1_1solution_1ab9d5af1574fa47670fe3e3a4caa323bb}

Creates an exact replica of this solution. The procedure to create a replica of a solution depends on the nature of the solution and must be defined in derived classes.

#### Returns
a solution instance that is equal to this solution.

#### `public bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` {#classdnn__opt_1_1core_1_1solution_1a7b7673c0c44f0f71be33314825154b0b}

Determines if the given object instance is assignable to this solution. This decision depends on the nature of the solution and must be defined in derived classes. Usually this comprobation must check if the given solution is of the same type, the same number of parameters.

#### Parameters
* `a` solution to check if it is assignable to this solution.





#### Returns
true if the given solution is the given solution is assignable, false otherwise.

#### `public inline shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > get_generator()` {#classdnn__opt_1_1core_1_1solution_1a99d9715a5f8b0ca3cef6a0967c29dab7}

get_generator returns a shared pointer containing a reference to the [parameter_generator](#classdnn__opt_1_1core_1_1parameter__generator) that is currently used to generate the paraters of this solution.

#### Returns
shared_ptr to the [parameter_generator](#classdnn__opt_1_1core_1_1parameter__generator) of this solution.

#### `protected bool _modified` {#classdnn__opt_1_1core_1_1solution_1a93f6d1847542079456ee8651e5522935}





#### `protected float _fitness` {#classdnn__opt_1_1core_1_1solution_1a08971743bd46a47ab8a3ab27c722f2eb}





#### `protected shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > _generator` {#classdnn__opt_1_1core_1_1solution_1ab2055f4e715c0bc9f1938485d4a501c1}





#### `protected inline  solution(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator)` {#classdnn__opt_1_1core_1_1solution_1ac324e6c6f9581bbe108cee96ec216121}

Default constructor of solution. It's protected so derived class should implement a factory pattern.



#### `protected float calculate_fitness()` {#classdnn__opt_1_1core_1_1solution_1af3dbd317aabae744d814c506577620fb}

Calculates the fitness of this solution. The nature of the fitness depends on the nature of the solution and must be implemented in derived classes.

#### Returns
the fitness of this solution.

# class `dnn_opt::core::solution_set` {#classdnn__opt_1_1core_1_1solution__set}


The [solution_set](#classdnn__opt_1_1core_1_1solution__set) class is intended to manage a set of optimization solutions for a determined optimization problem. This a helper class that can be usefull for population based optimization metaheuristics.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

June, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline int size()` | Returns the number of solutions of this container.
`public inline unique_ptr< `[`solution`](#classdnn__opt_1_1core_1_1solution)` > & get(int index)` | Returns an `unique_ptr< solution >` to a specified solution in this container.
`public inline void add(unique_ptr< `[`solution`](#classdnn__opt_1_1core_1_1solution)` > s)` | Appends a solution to the end of this container.
`public inline void set(int index,unique_ptr< `[`solution`](#classdnn__opt_1_1core_1_1solution)` > s)` | Changes a specific solution by a new one in this container.
`public inline void remove(int index)` | Removes a specified solution from this container.
`public inline float fitness()` | Calculates the average fitness of the solutions in this container.
`public inline void sort(bool lower_higher)` | Sorts the solutions of this container by its fitness.
`public inline void init()` | Initializes all solutions of this container by calling the init method of each of them.
`public inline unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > clone()` | Returns a copy to this container.
`protected inline  solution_set(int size)` | The basic constructor for a [solution_set](#classdnn__opt_1_1core_1_1solution__set).

## Members

#### `public inline int size()` {#classdnn__opt_1_1core_1_1solution__set_1af0a4c51de5f2343f49b3975a3abfe365}

Returns the number of solutions of this container.

#### Returns
the number of solutions of this container.

#### `public inline unique_ptr< `[`solution`](#classdnn__opt_1_1core_1_1solution)` > & get(int index)` {#classdnn__opt_1_1core_1_1solution__set_1a3d4cb995cd3e1ba803ce63ef407dbe36}

Returns an `unique_ptr< solution >` to a specified solution in this container.

#### Parameters
* `index` the index of a solution in this container.





#### Returns
an `unique_ptr< solution >` to the specified solution.

#### `public inline void add(unique_ptr< `[`solution`](#classdnn__opt_1_1core_1_1solution)` > s)` {#classdnn__opt_1_1core_1_1solution__set_1a13d1f5cfce0595d0e8a073c5ab432024}

Appends a solution to the end of this container.

#### Parameters
* `s` a `unique_ptr< solution >` that references to the solution.





#### Parameters
* `assertion` `if` the given solution is not assignable to the others.

#### `public inline void set(int index,unique_ptr< `[`solution`](#classdnn__opt_1_1core_1_1solution)` > s)` {#classdnn__opt_1_1core_1_1solution__set_1a4ef0f3eeeb9f82bcf2325ad4147f9935}

Changes a specific solution by a new one in this container.

#### Parameters
* `index` the index of the solution to be removed. 


* `s` a unique pointer that references to the new solution.





#### Parameters
* `assertion` `if` the given solution is not assignable to the others.

#### `public inline void remove(int index)` {#classdnn__opt_1_1core_1_1solution__set_1aea2f4dc7051473990c932e139830ddac}

Removes a specified solution from this container.

#### Parameters
* `index` the index of the solution to be removed.

#### `public inline float fitness()` {#classdnn__opt_1_1core_1_1solution__set_1ae8b128e39bc0ed5e795a1ce3b960d675}

Calculates the average fitness of the solutions in this container.

#### Returns
the average of fitness of the solutions.

#### `public inline void sort(bool lower_higher)` {#classdnn__opt_1_1core_1_1solution__set_1af7c5df3fb20f8aa97f9c2b7f29e4bd52}

Sorts the solutions of this container by its fitness.

#### Parameters
* `lower_higher` determines the ordering creiteria, `true` stands for lower to higher order and `false` stands for higher to lower order.

#### `public inline void init()` {#classdnn__opt_1_1core_1_1solution__set_1ad3169f96d1c5fbdd3f4c0b6f862d948a}

Initializes all solutions of this container by calling the init method of each of them.



#### `public inline unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > clone()` {#classdnn__opt_1_1core_1_1solution__set_1af6b8fe090d0e3e338e84856e979cc4a0}

Returns a copy to this container.

#### Returns
an `unique_ptr< [solution_set](#classdnn__opt_1_1core_1_1solution__set) >` which is a copy of this container.

#### `protected inline  solution_set(int size)` {#classdnn__opt_1_1core_1_1solution__set_1ae99e999896d01dceebf7e2cfcefdc6c1}

The basic constructor for a [solution_set](#classdnn__opt_1_1core_1_1solution__set).

#### Parameters
* `size` the number of solutions that this [solution_set](#classdnn__opt_1_1core_1_1solution__set) is going to manage. The [solution_set](#classdnn__opt_1_1core_1_1solution__set) can grow beyond this limit but settinng it at first can save time from re-allocation of memory.

# namespace `dnn_opt::core::activation_functions` {#namespacednn__opt_1_1core_1_1activation__functions}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`dnn_opt::core::activation_functions::elu`](#classdnn__opt_1_1core_1_1activation__functions_1_1elu)    | The elu class represents a elu function that can be used by an artificial neural network as activation function.
`class `[`dnn_opt::core::activation_functions::identity`](#classdnn__opt_1_1core_1_1activation__functions_1_1identity)    | The identity class represents a identity `x` function that can be used by an artificial neural network as activation function.
`class `[`dnn_opt::core::activation_functions::relu`](#classdnn__opt_1_1core_1_1activation__functions_1_1relu)    | The relu class represents a relu `max( 0, x )` function that can be used by an artificial neural network as activation function.
`class `[`dnn_opt::core::activation_functions::sigmoid`](#classdnn__opt_1_1core_1_1activation__functions_1_1sigmoid)    | The sigmoid class represents a sigmoid `1 / (1 + exp( - x ) )` function that can be used by an artificial neural network as activation function.
`class `[`dnn_opt::core::activation_functions::softmax`](#classdnn__opt_1_1core_1_1activation__functions_1_1softmax)    | The softmax class represents a softmax function that can be used by an artificial neural network as activation function.
`class `[`dnn_opt::core::activation_functions::tan_h`](#classdnn__opt_1_1core_1_1activation__functions_1_1tan__h)    | The [tan_h](#classdnn__opt_1_1core_1_1activation__functions_1_1tan__h) class represents a [tan_h](#classdnn__opt_1_1core_1_1activation__functions_1_1tan__h)`1 / (1 + exp( - x ) )` function that can be used by an artificial neural network as activation function.
# class `dnn_opt::core::activation_functions::elu` {#classdnn__opt_1_1core_1_1activation__functions_1_1elu}

```
class dnn_opt::core::activation_functions::elu
  : public dnn_opt::core::activation_function
```  

The elu class represents a elu function that can be used by an artificial neural network as activation function.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

September, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual float activation(const vector< float > & output,int index)` | activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

## Members

#### `public inline virtual float activation(const vector< float > & output,int index)` {#classdnn__opt_1_1core_1_1activation__functions_1_1elu_1a7d068b3ef6d941918c88d849f5e7b1bb}

activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

#### Parameters
* `output` `a` vector containing a layer output in terms of weighted summatory. 


* `index` `the` position of the corresponding unit in the layer which output is going to be calculated.





#### Returns
the activation value of the processing unit.

# class `dnn_opt::core::activation_functions::identity` {#classdnn__opt_1_1core_1_1activation__functions_1_1identity}

```
class dnn_opt::core::activation_functions::identity
  : public dnn_opt::core::activation_function
```  

The identity class represents a identity `x` function that can be used by an artificial neural network as activation function.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

September, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual float activation(const vector< float > & output,int index)` | activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

## Members

#### `public inline virtual float activation(const vector< float > & output,int index)` {#classdnn__opt_1_1core_1_1activation__functions_1_1identity_1a67ec440a99d296e9b6a9366581bf71fc}

activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

#### Parameters
* `output` `a` vector containing a layer output in terms of weighted summatory. 


* `index` `the` position of the corresponding unit in the layer which output is going to be calculated.





#### Returns
the activation value of the processing unit.

# class `dnn_opt::core::activation_functions::relu` {#classdnn__opt_1_1core_1_1activation__functions_1_1relu}

```
class dnn_opt::core::activation_functions::relu
  : public dnn_opt::core::activation_function
```  

The relu class represents a relu `max( 0, x )` function that can be used by an artificial neural network as activation function.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

September, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual float activation(const vector< float > & output,int index)` | activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

## Members

#### `public inline virtual float activation(const vector< float > & output,int index)` {#classdnn__opt_1_1core_1_1activation__functions_1_1relu_1a53875e14008f67fb9ce213588f81346b}

activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

#### Parameters
* `output` `a` vector containing a layer output in terms of weighted summatory. 


* `index` `the` position of the corresponding unit in the layer which output is going to be calculated.





#### Returns
the activation value of the processing unit.

# class `dnn_opt::core::activation_functions::sigmoid` {#classdnn__opt_1_1core_1_1activation__functions_1_1sigmoid}

```
class dnn_opt::core::activation_functions::sigmoid
  : public dnn_opt::core::activation_function
```  

The sigmoid class represents a sigmoid `1 / (1 + exp( - x ) )` function that can be used by an artificial neural network as activation function.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

September, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual float activation(const vector< float > & output,int index)` | activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

## Members

#### `public inline virtual float activation(const vector< float > & output,int index)` {#classdnn__opt_1_1core_1_1activation__functions_1_1sigmoid_1a74749818dbeed4d4265dab37a4b628d9}

activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

#### Parameters
* `output` `a` vector containing a layer output in terms of weighted summatory. 


* `index` `the` position of the corresponding unit in the layer which output is going to be calculated.





#### Returns
the activation value of the processing unit.

# class `dnn_opt::core::activation_functions::softmax` {#classdnn__opt_1_1core_1_1activation__functions_1_1softmax}

```
class dnn_opt::core::activation_functions::softmax
  : public dnn_opt::core::activation_function
```  

The softmax class represents a softmax function that can be used by an artificial neural network as activation function.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

September, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual float activation(const vector< float > & output,int index)` | activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

## Members

#### `public inline virtual float activation(const vector< float > & output,int index)` {#classdnn__opt_1_1core_1_1activation__functions_1_1softmax_1ab020f26d54acfcaca0f39a5c8f80f6ba}

activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

#### Parameters
* `output` `a` vector containing a layer output in terms of weighted summatory. 


* `index` `the` position of the corresponding unit in the layer which output is going to be calculated.





#### Returns
the activation value of the processing unit.

# class `dnn_opt::core::activation_functions::tan_h` {#classdnn__opt_1_1core_1_1activation__functions_1_1tan__h}

```
class dnn_opt::core::activation_functions::tan_h
  : public dnn_opt::core::activation_function
```  

The [tan_h](#classdnn__opt_1_1core_1_1activation__functions_1_1tan__h) class represents a [tan_h](#classdnn__opt_1_1core_1_1activation__functions_1_1tan__h)`1 / (1 + exp( - x ) )` function that can be used by an artificial neural network as activation function.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

September, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual float activation(const vector< float > & output,int index)` | activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

## Members

#### `public inline virtual float activation(const vector< float > & output,int index)` {#classdnn__opt_1_1core_1_1activation__functions_1_1tan__h_1a6c07d0111076d3d2a35ae62e460fe93b}

activation is a virtual method that have to be implemented by derived classes. Given the weighted summatory of an artificial neural network layer and the index of a specific unit it must return it's current activation value.

#### Parameters
* `output` `a` vector containing a layer output in terms of weighted summatory. 


* `index` `the` position of the corresponding unit in the layer which output is going to be calculated.





#### Returns
the activation value of the processing unit.

# namespace `dnn_opt::core::algorithms` {#namespacednn__opt_1_1core_1_1algorithms}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`dnn_opt::core::algorithms::cuckoo`](#classdnn__opt_1_1core_1_1algorithms_1_1cuckoo)    | The cuckoo class implements an optimization metaheuristic algorithm called Cuckoo Search (CS). This is a population based algorithm equiped with levy flights that allows an improved explotation and exploration of the search space.
`class `[`dnn_opt::core::algorithms::firefly`](#classdnn__opt_1_1core_1_1algorithms_1_1firefly)    | The firefly class implements an optimization metaheuristic algorithm called Firefly Algorithm (FA). This is a population based algorithm inspired in the bio-luminicence of fireflies.
`class `[`dnn_opt::core::algorithms::gray_wolf`](#classdnn__opt_1_1core_1_1algorithms_1_1gray__wolf)    | The [gray_wolf](#classdnn__opt_1_1core_1_1algorithms_1_1gray__wolf) class implements an optimization metaheuristic algorithm called Gray Wolf Optimizer (GWO). This is a population based algorithm inspired in the hunting procedure of gray wolfs.
`class `[`dnn_opt::core::algorithms::pso`](#classdnn__opt_1_1core_1_1algorithms_1_1pso)    | The pso class implements an optimization metaheuristic algorithm called Particle Swarm Optimization (PSO). This is a population based algorithm inspired in the movements of swarms.
# class `dnn_opt::core::algorithms::cuckoo` {#classdnn__opt_1_1core_1_1algorithms_1_1cuckoo}

```
class dnn_opt::core::algorithms::cuckoo
  : public dnn_opt::core::algorithm
```  

The cuckoo class implements an optimization metaheuristic algorithm called Cuckoo Search (CS). This is a population based algorithm equiped with levy flights that allows an improved explotation and exploration of the search space.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

July, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void optimize()` | Performs a single steep of optimization for this algorithm.
`protected inline  cuckoo(float scale,float levy,float replacement,unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > solutions,bool maximization)` | Creates a new instance of the Cuckoo Search algorithm. This constructor is protector and derived classes should implement a factory pattern.

## Members

#### `public inline virtual void optimize()` {#classdnn__opt_1_1core_1_1algorithms_1_1cuckoo_1a93e131798f45797a437ceb736b086690}

Performs a single steep of optimization for this algorithm.



#### `protected inline  cuckoo(float scale,float levy,float replacement,unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > solutions,bool maximization)` {#classdnn__opt_1_1core_1_1algorithms_1_1cuckoo_1a4222dc46bfad56c116a3a376ee49b970}

Creates a new instance of the Cuckoo Search algorithm. This constructor is protector and derived classes should implement a factory pattern.

#### Parameters
* `scale` is the scale of the search space. 


* `levy` is the Levy parameter for the levy's walk. 


* `replacement` is the proportion to replace bad solutions. 


* `solutions` is the set of candidate solutions.

# class `dnn_opt::core::algorithms::firefly` {#classdnn__opt_1_1core_1_1algorithms_1_1firefly}

```
class dnn_opt::core::algorithms::firefly
  : public dnn_opt::core::algorithm
```  

The firefly class implements an optimization metaheuristic algorithm called Firefly Algorithm (FA). This is a population based algorithm inspired in the bio-luminicence of fireflies.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

July, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void optimize()` | Performs a single steep of optimization for this algorithm.
`protected inline  firefly(float light_decay,float rand_influence,float init_bright,unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > solutions,bool maximization)` | Creates a new instance of the Firefly Algorithm. Derived classes should implement factory pattern.

## Members

#### `public inline virtual void optimize()` {#classdnn__opt_1_1core_1_1algorithms_1_1firefly_1ab9cd140330f973ef99ea028e0ffb6d96}

Performs a single steep of optimization for this algorithm.



#### `protected inline  firefly(float light_decay,float rand_influence,float init_bright,unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > solutions,bool maximization)` {#classdnn__opt_1_1core_1_1algorithms_1_1firefly_1a4f1ca0365f4890f59287c2ffb34064ba}

Creates a new instance of the Firefly Algorithm. Derived classes should implement factory pattern.

#### Parameters
* `light_decay` the absorption of light by the space. 


* `rand_influence` the influence of randomess in the problem. 


* `init_bright` the bright of a firefly light when distance is cero. 


* `solutions` a set of candidate solutions to optimize. 


* `maximization` if true this is a maximization problem, if false a minimization problem. Default is true.

# class `dnn_opt::core::algorithms::gray_wolf` {#classdnn__opt_1_1core_1_1algorithms_1_1gray__wolf}

```
class dnn_opt::core::algorithms::gray_wolf
  : public dnn_opt::core::algorithm
```  

The [gray_wolf](#classdnn__opt_1_1core_1_1algorithms_1_1gray__wolf) class implements an optimization metaheuristic algorithm called Gray Wolf Optimizer (GWO). This is a population based algorithm inspired in the hunting procedure of gray wolfs.

## References



* MIRJALILI, Seyedali; MIRJALILI, Seyed Mohammad; LEWIS, Andrew. Grey wolf optimizer. Advances in Engineering Software, 2014, vol. 69, p. 46-61.





Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

July, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void optimize()` | Performs a single steep of optimization for this algorithm.
`protected inline  gray_wolf(float decrease_factor,unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > solutions)` | The basic contructor for a [gray_wolf](#classdnn__opt_1_1core_1_1algorithms_1_1gray__wolf) optimization class.

## Members

#### `public inline virtual void optimize()` {#classdnn__opt_1_1core_1_1algorithms_1_1gray__wolf_1a34153c93428698feecfbd8b044d1645e}

Performs a single steep of optimization for this algorithm.



#### `protected inline  gray_wolf(float decrease_factor,unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > solutions)` {#classdnn__opt_1_1core_1_1algorithms_1_1gray__wolf_1a3fcaeeaaeb5a7a88ff0e6dabfa33dfb4}

The basic contructor for a [gray_wolf](#classdnn__opt_1_1core_1_1algorithms_1_1gray__wolf) optimization class.

#### Parameters
* `decrease_factor` a parameters that controls the  parameter of the original algorithm. This parameter have to be tunned according to the expected number of optimization steeps. 


* `solutions` a set of individuals.





#### Parameters
* `invalid_argument` `if` the given [solution_set](#classdnn__opt_1_1core_1_1solution__set) does not have at least one solution to optimize.

# class `dnn_opt::core::algorithms::pso` {#classdnn__opt_1_1core_1_1algorithms_1_1pso}

```
class dnn_opt::core::algorithms::pso
  : public dnn_opt::core::algorithm
```  

The pso class implements an optimization metaheuristic algorithm called Particle Swarm Optimization (PSO). This is a population based algorithm inspired in the movements of swarms.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

1.0 

July, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void optimize()` | Performs a single steep of optimization for this algorithm.
`protected inline  pso(float local_param,float global_param,float speed_param,unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > solutions,bool maximization)` | The basic contructor of a pso class.

## Members

#### `public inline virtual void optimize()` {#classdnn__opt_1_1core_1_1algorithms_1_1pso_1a617aac6bccc53122577ae655691da9ab}

Performs a single steep of optimization for this algorithm.



#### `protected inline  pso(float local_param,float global_param,float speed_param,unique_ptr< `[`solution_set`](#classdnn__opt_1_1core_1_1solution__set)` > solutions,bool maximization)` {#classdnn__opt_1_1core_1_1algorithms_1_1pso_1ae828603c734f62704da6df900fef1cae}

The basic contructor of a pso class.

#### Parameters
* `local_param` the contribution of local best solutions. 


* `global_param` the contribution of the global best solution. 


* `speed_param` the contribution of the speed in each particle movement. 


* `solutions` a set of individuals.

# namespace `dnn_opt::core::error_functions` {#namespacednn__opt_1_1core_1_1error__functions}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`dnn_opt::core::error_functions::mse`](#classdnn__opt_1_1core_1_1error__functions_1_1mse)    | 
`class `[`dnn_opt::core::error_functions::overall_error`](#classdnn__opt_1_1core_1_1error__functions_1_1overall__error)    | 
# class `dnn_opt::core::error_functions::mse` {#classdnn__opt_1_1core_1_1error__functions_1_1mse}

```
class dnn_opt::core::error_functions::mse
  : public dnn_opt::core::error_function
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual float error(const vector< vector< float > > & real,const vector< vector< float > > & expected)` | Calculates the error value between the real output of an artificial neural network and an expected output.

## Members

#### `public inline virtual float error(const vector< vector< float > > & real,const vector< vector< float > > & expected)` {#classdnn__opt_1_1core_1_1error__functions_1_1mse_1a68c7188d3516550577e5281dfdfddae5}

Calculates the error value between the real output of an artificial neural network and an expected output.

#### Parameters
* `real` a multi-target output of an artificial neural network resulting of the propagation of several training / validation patterns.


* `expected` a multi-target expected output of an artificial neural network.





#### Returns
the error value between the real output of the network and the expected output.


#### Parameters
* `assertion` if `real.size() != expected.size()`

# class `dnn_opt::core::error_functions::overall_error` {#classdnn__opt_1_1core_1_1error__functions_1_1overall__error}

```
class dnn_opt::core::error_functions::overall_error
  : public dnn_opt::core::error_function
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual float error(const vector< vector< float > > & real,const vector< vector< float > > & expected)` | Calculates the error value between the real output of an artificial neural network and an expected output.

## Members

#### `public inline virtual float error(const vector< vector< float > > & real,const vector< vector< float > > & expected)` {#classdnn__opt_1_1core_1_1error__functions_1_1overall__error_1aa6665ef10b549da4bc5a4eca9e31f219}

Calculates the error value between the real output of an artificial neural network and an expected output.

#### Parameters
* `real` a multi-target output of an artificial neural network resulting of the propagation of several training / validation patterns.


* `expected` a multi-target expected output of an artificial neural network.





#### Returns
the error value between the real output of the network and the expected output.


#### Parameters
* `assertion` if `real.size() != expected.size()`

# namespace `dnn_opt::core::io` {#namespacednn__opt_1_1core_1_1io}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`dnn_opt::core::io::file_reader`](#classdnn__opt_1_1core_1_1io_1_1file__reader)    | This class is intended to fetch training patterns from file. The file must have the following structure:
# class `dnn_opt::core::io::file_reader` {#classdnn__opt_1_1core_1_1io_1_1file__reader}

```
class dnn_opt::core::io::file_reader
  : public dnn_opt::core::reader
```  

This class is intended to fetch training patterns from file. The file must have the following structure:

* In the first line two integers separated by a space, the input dimension `n` and the output dimension `m`.


* In the following lines, each line represents a pattern containing `n` doubles for the input followed by `m` doubles representing the expected output.





Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

1.0 

June, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual vector< vector< float > > & get_input_data()` | Returns a vector containing all input patterns of the file.
`public inline virtual vector< vector< float > > & get_output_data()` | Returns a vector containing all output patterns of the file.
`protected inline  file_reader(string file_name)` | The basic contructor for [file_reader](#classdnn__opt_1_1core_1_1io_1_1file__reader) class.

## Members

#### `public inline virtual vector< vector< float > > & get_input_data()` {#classdnn__opt_1_1core_1_1io_1_1file__reader_1a94ea97a3c2869a94d52a790fa5a7902b}

Returns a vector containing all input patterns of the file.

#### Returns
a vector with the input patterns.

#### `public inline virtual vector< vector< float > > & get_output_data()` {#classdnn__opt_1_1core_1_1io_1_1file__reader_1a75be18c714ea96860d1e6b531fef5605}

Returns a vector containing all output patterns of the file.

#### Returns
a vector of output patterns.

#### `protected inline  file_reader(string file_name)` {#classdnn__opt_1_1core_1_1io_1_1file__reader_1a78a0aa2728222dba27fd5ffee20158c6}

The basic contructor for [file_reader](#classdnn__opt_1_1core_1_1io_1_1file__reader) class.

#### Parameters
* `file_name` the file location of the training database file.





#### Parameters
* `assertion` if the file_name provided is incorrect

# namespace `dnn_opt::core::layers` {#namespacednn__opt_1_1core_1_1layers}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`dnn_opt::core::layers::convolutional`](#classdnn__opt_1_1core_1_1layers_1_1convolutional)    | 
`class `[`dnn_opt::core::layers::discretization`](#classdnn__opt_1_1core_1_1layers_1_1discretization)    | 
`class `[`dnn_opt::core::layers::fully_connected`](#classdnn__opt_1_1core_1_1layers_1_1fully__connected)    | The fully_connected_layer class represents a layer of processing units of an artificial neural network where each unit is fully connected to the output of the previous layer. When considering the layer parameters, those are arranged in a consecutive way such as: a unit weights and bias term are followed by the next unit's weights and bias. Layer's parameters are provided externally, hence this class is intended to provide only an add-hoc feature for ann_pt::solution derived classes.
`class `[`dnn_opt::core::layers::max_pooling`](#classdnn__opt_1_1core_1_1layers_1_1max__pooling)    | 
# class `dnn_opt::core::layers::convolutional` {#classdnn__opt_1_1core_1_1layers_1_1convolutional}

```
class dnn_opt::core::layers::convolutional
  : public dnn_opt::core::layer
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void propagate(const vector< float > & input,const vector< float > & params,int start,int end)` | propagate is an interface to propagate a given input signal to an output. The output value is managed by this class, hence to obtain the current output value resulting of calling this method you should call
`public inline virtual int count() const` | count returns the number of parameters that is required by this layer.
`public inline virtual const vector< float > & output() const` | output returns a vector vith the output values of this layer after
`public inline virtual `[`layer`](#classdnn__opt_1_1core_1_1layer)` * clone()` | clone returns an exact copy of this layer.
`protected inline  convolutional(int in_height,int in_width,int in_depth,int w_height,int w_width,int kernel_count,int padding,int stride,vector< vector< bool > > connection_table,shared_ptr< `[`activation_function`](#classdnn__opt_1_1core_1_1activation__function)` > AF)` | convolutional creates a convolutional layer with the specified parameters and parameter sharing. A convolutional layer expects to be provided with 3D inputs

## Members

#### `public inline virtual void propagate(const vector< float > & input,const vector< float > & params,int start,int end)` {#classdnn__opt_1_1core_1_1layers_1_1convolutional_1a0c718ac60e713c8fec4cbd18cbf56319}

propagate is an interface to propagate a given input signal to an output. The output value is managed by this class, hence to obtain the current output value resulting of calling this method you should call

**See also**: [output()](#classdnn__opt_1_1core_1_1layers_1_1convolutional_1a148ecc82dad77824263128fe9972c538). Derived classes are encouraged to call this method for validation.

**Warning:** the number of values in the input vector should ve equal to 
**See also**: [input_dimension()](#classdnn__opt_1_1core_1_1layer_1ab502eb1b011bdf8008e6301665d789a0). At the same time start and end must be valid indexes in the params vector, start <= end. Failing these requirements produces assertion error.


#### Parameters
* `input` is the input signal to be propagated. 


* `params` is a vector containing a parameter list to be used for this layer. 


* `start` is where the parameters of this layer starts in the params vector. 


* `end` is where the paramters of this layer ends in the params vector.





#### Parameters
* `assertion` if any of the previously mentioned conditions fails.

#### `public inline virtual int count() const` {#classdnn__opt_1_1core_1_1layers_1_1convolutional_1adf3dfc71272c4e87c616ac71cca49140}

count returns the number of parameters that is required by this layer.

#### Returns
the number of parameters.

#### `public inline virtual const vector< float > & output() const` {#classdnn__opt_1_1core_1_1layers_1_1convolutional_1a148ecc82dad77824263128fe9972c538}

output returns a vector vith the output values of this layer after

**See also**: [propagate()](#classdnn__opt_1_1core_1_1layers_1_1convolutional_1a0c718ac60e713c8fec4cbd18cbf56319) was called. Calling this method before 


**See also**: [propagate()](#classdnn__opt_1_1core_1_1layers_1_1convolutional_1a0c718ac60e713c8fec4cbd18cbf56319) produces inconsistent results.


#### Returns
a vector with 


**See also**: [output_dimension()](#classdnn__opt_1_1core_1_1layer_1a64063d73e577923444900256512d364c) values corresponging to the [output](#classdnn__opt_1_1core_1_1layers_1_1convolutional_1a148ecc82dad77824263128fe9972c538) of this [layer](#classdnn__opt_1_1core_1_1layer).

#### `public inline virtual `[`layer`](#classdnn__opt_1_1core_1_1layer)` * clone()` {#classdnn__opt_1_1core_1_1layers_1_1convolutional_1a3cd5f16c46e026685f47259ea46eba58}

clone returns an exact copy of this layer.

#### Returns
a pointer to the copyy of this layer.

#### `protected inline  convolutional(int in_height,int in_width,int in_depth,int w_height,int w_width,int kernel_count,int padding,int stride,vector< vector< bool > > connection_table,shared_ptr< `[`activation_function`](#classdnn__opt_1_1core_1_1activation__function)` > AF)` {#classdnn__opt_1_1core_1_1layers_1_1convolutional_1adaf571af26f973ba67eb00d6725d5b75}

convolutional creates a convolutional layer with the specified parameters and parameter sharing. A convolutional layer expects to be provided with 3D inputs

**See also**: [propagate](#classdnn__opt_1_1core_1_1layers_1_1convolutional_1a0c718ac60e713c8fec4cbd18cbf56319) in the following way:      (1) depth column of (1,1) followed by,
     (2) depth column of (1,2) until,
     (3) depth column of (1, in_width) and then,
     (4) depth column of (in_height, in_width)



#### Parameters
* `in_height` the height of the input volume. 


* `in_width` the width of the input volume. 


* `in_depth` the depth of the input volume. 


* `w_height` the height of a kernel. 


* `w_width` the width of a kernel. 


* `kernel_count` the number of kernels. 


* `padding` the number of ceros added to the border of the input volume. 


* `stride` the number of steeps used for kerners to slide the convolution window. 


* `AF` the activation function of this layer.





#### Parameters
* `assertion` `if` the kernel window can not be stridded in the width/height dimension of the input volume.

# class `dnn_opt::core::layers::discretization` {#classdnn__opt_1_1core_1_1layers_1_1discretization}

```
class dnn_opt::core::layers::discretization
  : public dnn_opt::core::layer
```  



: Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

October, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void propagate(const vector< float > & input,const vector< float > & params,int start,int end)` | propagate is an interface to propagate a given input signal to an output. The output value is managed by this class, hence to obtain the current output value resulting of calling this method you should call
`public inline virtual int count() const` | count returns the number of parameters that is required by this layer.
`public inline virtual const vector< float > & output() const` | output returns a vector vith the output values of this layer after
`public inline virtual `[`layer`](#classdnn__opt_1_1core_1_1layer)` * clone()` | clone returns an exact copy of this layer.
`protected vector< float > _output` | 
`protected function< vector< float > vector< float >) > _criteria` | 
`protected inline  discretization(int in_dim,function< vector< float >(vector< float >) > criteria)` | 

## Members

#### `public inline virtual void propagate(const vector< float > & input,const vector< float > & params,int start,int end)` {#classdnn__opt_1_1core_1_1layers_1_1discretization_1a7e9209ea57e8ec6fc932af5fff5d8a85}

propagate is an interface to propagate a given input signal to an output. The output value is managed by this class, hence to obtain the current output value resulting of calling this method you should call

**See also**: [output()](#classdnn__opt_1_1core_1_1layers_1_1discretization_1a3bb888a43f9498d6876729528f89861d). Derived classes are encouraged to call this method for validation.

**Warning:** the number of values in the input vector should ve equal to 
**See also**: [input_dimension()](#classdnn__opt_1_1core_1_1layer_1ab502eb1b011bdf8008e6301665d789a0). At the same time start and end must be valid indexes in the params vector, start <= end. Failing these requirements produces assertion error.


#### Parameters
* `input` is the input signal to be propagated. 


* `params` is a vector containing a parameter list to be used for this layer. 


* `start` is where the parameters of this layer starts in the params vector. 


* `end` is where the paramters of this layer ends in the params vector.





#### Parameters
* `assertion` if any of the previously mentioned conditions fails.

#### `public inline virtual int count() const` {#classdnn__opt_1_1core_1_1layers_1_1discretization_1a49f12f00b862c641fcf23b5ace6cf2fc}

count returns the number of parameters that is required by this layer.

#### Returns
the number of parameters.

#### `public inline virtual const vector< float > & output() const` {#classdnn__opt_1_1core_1_1layers_1_1discretization_1a3bb888a43f9498d6876729528f89861d}

output returns a vector vith the output values of this layer after

**See also**: [propagate()](#classdnn__opt_1_1core_1_1layers_1_1discretization_1a7e9209ea57e8ec6fc932af5fff5d8a85) was called. Calling this method before 


**See also**: [propagate()](#classdnn__opt_1_1core_1_1layers_1_1discretization_1a7e9209ea57e8ec6fc932af5fff5d8a85) produces inconsistent results.


#### Returns
a vector with 


**See also**: [output_dimension()](#classdnn__opt_1_1core_1_1layer_1a64063d73e577923444900256512d364c) values corresponging to the [output](#classdnn__opt_1_1core_1_1layers_1_1discretization_1a3bb888a43f9498d6876729528f89861d) of this [layer](#classdnn__opt_1_1core_1_1layer).

#### `public inline virtual `[`layer`](#classdnn__opt_1_1core_1_1layer)` * clone()` {#classdnn__opt_1_1core_1_1layers_1_1discretization_1ac7e103a8dd2e8ffc41680d5b02ece9e2}

clone returns an exact copy of this layer.

#### Returns
a pointer to the copyy of this layer.

#### `protected vector< float > _output` {#classdnn__opt_1_1core_1_1layers_1_1discretization_1a2bf94e55659688703afda2c635e2888e}





#### `protected function< vector< float > vector< float >) > _criteria` {#classdnn__opt_1_1core_1_1layers_1_1discretization_1ac6fc5bc61208521238369f35eac3f664}





#### `protected inline  discretization(int in_dim,function< vector< float >(vector< float >) > criteria)` {#classdnn__opt_1_1core_1_1layers_1_1discretization_1aecbbd6f9c0f2c97a47241f093354b5e8}





# class `dnn_opt::core::layers::fully_connected` {#classdnn__opt_1_1core_1_1layers_1_1fully__connected}

```
class dnn_opt::core::layers::fully_connected
  : public dnn_opt::core::layer
```  

The fully_connected_layer class represents a layer of processing units of an artificial neural network where each unit is fully connected to the output of the previous layer. When considering the layer parameters, those are arranged in a consecutive way such as: a unit weights and bias term are followed by the next unit's weights and bias. Layer's parameters are provided externally, hence this class is intended to provide only an add-hoc feature for ann_pt::solution derived classes.

: Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

September, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void propagate(const vector< float > & input,const vector< float > & params,int start,int end)` | propagate a given input signal through the layer by calculating the multiplication of each input signal by the unit's parameters.
`public inline virtual int count() const` | count returns the number of parameters that is required by this layer. This is [input_dimension()](#classdnn__opt_1_1core_1_1layer_1ab502eb1b011bdf8008e6301665d789a0) * [output_dimension()](#classdnn__opt_1_1core_1_1layer_1a64063d73e577923444900256512d364c) since this is a fully connected layer.
`public inline virtual const vector< float > & output() const` | output returns a vector vith the output values of this layer after
`public inline virtual `[`layer`](#classdnn__opt_1_1core_1_1layer)` * clone()` | clone returns an exact copy of this layer.
`protected vector< float > _output` | 
`protected vector< float > _w_summ` | 
`protected inline  fully_connected(int in_dim,int out_dim,shared_ptr< `[`activation_function`](#classdnn__opt_1_1core_1_1activation__function)` > AF)` | fully_connected_layer creates a fully_connected_layer instance.

## Members

#### `public inline virtual void propagate(const vector< float > & input,const vector< float > & params,int start,int end)` {#classdnn__opt_1_1core_1_1layers_1_1fully__connected_1a3b8adda7f14442ba32b59c9aa0b22519}

propagate a given input signal through the layer by calculating the multiplication of each input signal by the unit's parameters.

#### Parameters
* `input` `a` vector containing the input signal to be propagated. 


* `param_begin` `an` iterator pointing to the begining of a container with the layer parameters. 


* `param_end` `an` iterator pointing to the end of this layer parameters in a container.





#### Returns
a vector with the ouput values of this layer.

#### `public inline virtual int count() const` {#classdnn__opt_1_1core_1_1layers_1_1fully__connected_1a9e1f2e6b74674a513f2a7dc199b3e8ef}

count returns the number of parameters that is required by this layer. This is [input_dimension()](#classdnn__opt_1_1core_1_1layer_1ab502eb1b011bdf8008e6301665d789a0) * [output_dimension()](#classdnn__opt_1_1core_1_1layer_1a64063d73e577923444900256512d364c) since this is a fully connected layer.

#### Returns
the number of parameters.

#### `public inline virtual const vector< float > & output() const` {#classdnn__opt_1_1core_1_1layers_1_1fully__connected_1a2c04e03fcc8929da1d66946fe252966b}

output returns a vector vith the output values of this layer after

**See also**: [propagate()](#classdnn__opt_1_1core_1_1layers_1_1fully__connected_1a3b8adda7f14442ba32b59c9aa0b22519) was called. Calling this method before 


**See also**: [propagate()](#classdnn__opt_1_1core_1_1layers_1_1fully__connected_1a3b8adda7f14442ba32b59c9aa0b22519) produces inconsistent results.


#### Returns
a vector with 


**See also**: [output_dimension()](#classdnn__opt_1_1core_1_1layer_1a64063d73e577923444900256512d364c) values corresponging to the [output](#classdnn__opt_1_1core_1_1layers_1_1fully__connected_1a2c04e03fcc8929da1d66946fe252966b) of this [layer](#classdnn__opt_1_1core_1_1layer).

#### `public inline virtual `[`layer`](#classdnn__opt_1_1core_1_1layer)` * clone()` {#classdnn__opt_1_1core_1_1layers_1_1fully__connected_1a6b925ba4ed4447a1b9288bec702e7619}

clone returns an exact copy of this layer.

#### Returns
a pointer to the copyy of this layer.

#### `protected vector< float > _output` {#classdnn__opt_1_1core_1_1layers_1_1fully__connected_1af06215ccc90be00157fbfe71c220c820}





#### `protected vector< float > _w_summ` {#classdnn__opt_1_1core_1_1layers_1_1fully__connected_1abc14db9836a83a16bc67ac859debbea5}





#### `protected inline  fully_connected(int in_dim,int out_dim,shared_ptr< `[`activation_function`](#classdnn__opt_1_1core_1_1activation__function)` > AF)` {#classdnn__opt_1_1core_1_1layers_1_1fully__connected_1a9703f6483e7e2c63e3db6fe73a015d6a}

fully_connected_layer creates a fully_connected_layer instance.

#### Parameters
* `input_dimension` the number of input dimensions. 


* `output_dimension` the number of output dimensions.

# class `dnn_opt::core::layers::max_pooling` {#classdnn__opt_1_1core_1_1layers_1_1max__pooling}

```
class dnn_opt::core::layers::max_pooling
  : public dnn_opt::core::layer
```  



: Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

October, 2016 

1.0

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void propagate(const vector< float > & input,const vector< float > & params,int start,int end)` | propagate is an interface to propagate a given input signal to an output. The output value is managed by this class, hence to obtain the current output value resulting of calling this method you should call
`public inline virtual int count() const` | count returns the number of parameters that is required by this layer. This is [input_dimension()](#classdnn__opt_1_1core_1_1layer_1ab502eb1b011bdf8008e6301665d789a0) * [output_dimension()](#classdnn__opt_1_1core_1_1layer_1a64063d73e577923444900256512d364c) since this is a fully connected layer.
`public inline virtual const vector< float > & output() const` | output returns a vector vith the output values of this layer after
`public inline virtual `[`layer`](#classdnn__opt_1_1core_1_1layer)` * clone()` | clone returns an exact copy of this layer.
`protected inline  max_pooling(int in_height,int in_width,int in_depth,int w_height,int w_width,int stride)` | 

## Members

#### `public inline virtual void propagate(const vector< float > & input,const vector< float > & params,int start,int end)` {#classdnn__opt_1_1core_1_1layers_1_1max__pooling_1a828d3fff8e242252e0f142e2e75de83a}

propagate is an interface to propagate a given input signal to an output. The output value is managed by this class, hence to obtain the current output value resulting of calling this method you should call

**See also**: [output()](#classdnn__opt_1_1core_1_1layers_1_1max__pooling_1a3641ee789013110d1a7287b26cfc86d2). Derived classes are encouraged to call this method for validation.

**Warning:** the number of values in the input vector should ve equal to 
**See also**: [input_dimension()](#classdnn__opt_1_1core_1_1layer_1ab502eb1b011bdf8008e6301665d789a0). At the same time start and end must be valid indexes in the params vector, start <= end. Failing these requirements produces assertion error.


#### Parameters
* `input` is the input signal to be propagated. 


* `params` is a vector containing a parameter list to be used for this layer. 


* `start` is where the parameters of this layer starts in the params vector. 


* `end` is where the paramters of this layer ends in the params vector.





#### Parameters
* `assertion` if any of the previously mentioned conditions fails.

#### `public inline virtual int count() const` {#classdnn__opt_1_1core_1_1layers_1_1max__pooling_1a4cdc774855312a45fed70b069161c269}

count returns the number of parameters that is required by this layer. This is [input_dimension()](#classdnn__opt_1_1core_1_1layer_1ab502eb1b011bdf8008e6301665d789a0) * [output_dimension()](#classdnn__opt_1_1core_1_1layer_1a64063d73e577923444900256512d364c) since this is a fully connected layer.

#### Returns
the number of parameters.

#### `public inline virtual const vector< float > & output() const` {#classdnn__opt_1_1core_1_1layers_1_1max__pooling_1a3641ee789013110d1a7287b26cfc86d2}

output returns a vector vith the output values of this layer after

**See also**: [propagate()](#classdnn__opt_1_1core_1_1layers_1_1max__pooling_1a828d3fff8e242252e0f142e2e75de83a) was called. Calling this method before 


**See also**: [propagate()](#classdnn__opt_1_1core_1_1layers_1_1max__pooling_1a828d3fff8e242252e0f142e2e75de83a) produces inconsistent results.


#### Returns
a vector with 


**See also**: [output_dimension()](#classdnn__opt_1_1core_1_1layer_1a64063d73e577923444900256512d364c) values corresponging to the [output](#classdnn__opt_1_1core_1_1layers_1_1max__pooling_1a3641ee789013110d1a7287b26cfc86d2) of this [layer](#classdnn__opt_1_1core_1_1layer).

#### `public inline virtual `[`layer`](#classdnn__opt_1_1core_1_1layer)` * clone()` {#classdnn__opt_1_1core_1_1layers_1_1max__pooling_1a4116540d3bf157f57e2549ca7ffb3c5f}

clone returns an exact copy of this layer.

#### Returns
a pointer to the copyy of this layer.

#### `protected inline  max_pooling(int in_height,int in_width,int in_depth,int w_height,int w_width,int stride)` {#classdnn__opt_1_1core_1_1layers_1_1max__pooling_1a7e2f640c18433d2af24c7f7a400c7f21}





# namespace `dnn_opt::core::parameter_generators` {#namespacednn__opt_1_1core_1_1parameter__generators}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`dnn_opt::core::parameter_generators::normal`](#classdnn__opt_1_1core_1_1parameter__generators_1_1normal)    | 
`class `[`dnn_opt::core::parameter_generators::uniform`](#classdnn__opt_1_1core_1_1parameter__generators_1_1uniform)    | 
# class `dnn_opt::core::parameter_generators::normal` {#classdnn__opt_1_1core_1_1parameter__generators_1_1normal}

```
class dnn_opt::core::parameter_generators::normal
  : public dnn_opt::core::parameter_generator
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual float generate()` | generate a single parameter.
`protected inline  normal(float mean,float dev)` | 

## Members

#### `public inline virtual float generate()` {#classdnn__opt_1_1core_1_1parameter__generators_1_1normal_1ac766f703c362c552367554c0727056e0}

generate a single parameter.

#### Returns
the value of the parameter.

#### `protected inline  normal(float mean,float dev)` {#classdnn__opt_1_1core_1_1parameter__generators_1_1normal_1a6b7744dc2bf1a212a1c15bc753f773f2}





# class `dnn_opt::core::parameter_generators::uniform` {#classdnn__opt_1_1core_1_1parameter__generators_1_1uniform}

```
class dnn_opt::core::parameter_generators::uniform
  : public dnn_opt::core::parameter_generator
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual float generate()` | generate a single parameter.
`protected inline  uniform(float min,float max)` | 

## Members

#### `public inline virtual float generate()` {#classdnn__opt_1_1core_1_1parameter__generators_1_1uniform_1af36b0de5e6b130432411bb90aae58295}

generate a single parameter.

#### Returns
the value of the parameter.

#### `protected inline  uniform(float min,float max)` {#classdnn__opt_1_1core_1_1parameter__generators_1_1uniform_1a80f0a7da076b7842f09a1c99bea61b25}





# namespace `dnn_opt::core::solutions` {#namespacednn__opt_1_1core_1_1solutions}



## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`class `[`dnn_opt::core::solutions::ackley`](#classdnn__opt_1_1core_1_1solutions_1_1ackley)    | The ackley class represents an optimization solutions which fitness cost is calculated via Ackley function.
`class `[`dnn_opt::core::solutions::de_jung`](#classdnn__opt_1_1core_1_1solutions_1_1de__jung)    | The [de_jung](#classdnn__opt_1_1core_1_1solutions_1_1de__jung) class represents an optimization solutions which fitness cost is calculated via De' Jung function.
`class `[`dnn_opt::core::solutions::griewangk`](#classdnn__opt_1_1core_1_1solutions_1_1griewangk)    | The griewangk class.
`class `[`dnn_opt::core::solutions::michalewicz`](#classdnn__opt_1_1core_1_1solutions_1_1michalewicz)    | The michalewicz class.
`class `[`dnn_opt::core::solutions::network`](#classdnn__opt_1_1core_1_1solutions_1_1network)    | 
`class `[`dnn_opt::core::solutions::rastrigin`](#classdnn__opt_1_1core_1_1solutions_1_1rastrigin)    | The rastrigin class represents an optimization solutions which fitness cost is calculated via Rastrigin function.
`class `[`dnn_opt::core::solutions::rosenbrock`](#classdnn__opt_1_1core_1_1solutions_1_1rosenbrock)    | The rosenbrock class represents an optimization solutions which fitness cost is calculated via Rosenbrock function.
`class `[`dnn_opt::core::solutions::schwefel`](#classdnn__opt_1_1core_1_1solutions_1_1schwefel)    | The schwefel class.
`class `[`dnn_opt::core::solutions::styblinski_tang`](#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang)    | The [styblinski_tang](#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang) class represents an optimization solutions which fitness cost is calculated via Styblinski-Tang function.
# class `dnn_opt::core::solutions::ackley` {#classdnn__opt_1_1core_1_1solutions_1_1ackley}

```
class dnn_opt::core::solutions::ackley
  : public dnn_opt::core::solution
```  

The ackley class represents an optimization solutions which fitness cost is calculated via Ackley function.

Ackley function have a global minima in {0,..., 0} with a value of 0. A commonly used search domain for testing is [-5, 5].

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

1.0 

November, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void set(int index,float value)` | Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.
`public inline virtual float get(int index) const` | Returns the value of a given parameter.
`public inline virtual int size() const` | Returns the number of parameters of this solution.
`public inline virtual vector< float > & get_parameters()` | Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.
`public inline virtual `[`ackley`](#classdnn__opt_1_1core_1_1solutions_1_1ackley)` * clone()` | Creates an exact replica of this solution.
`public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` | Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an ackley solution and have the same number of parameters.
`protected vector< float > _params` | 
`protected inline  ackley(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` | This is the basic contructor for this class. Is protected since this this class implements the factory pattern. Derived clasess however can use this constructor.
`protected inline virtual float calculate_fitness()` | Calculates the fitness of this solution which in this case is determined by the Ackley function. More information about Ackley function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

## Members

#### `public inline virtual void set(int index,float value)` {#classdnn__opt_1_1core_1_1solutions_1_1ackley_1a1e1a652b8d267a81f65c7502951b6b8e}

Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.

#### Parameters
* `index` the index of the parameter to be changed. 


* `value` the new parameter's value.

#### `public inline virtual float get(int index) const` {#classdnn__opt_1_1core_1_1solutions_1_1ackley_1ac2c1c0d58c8e03c72ae642a0bc4e7eb5}

Returns the value of a given parameter.

#### Parameters
* `index` the index of the parameter to be retorned.





#### Returns
the current value of the parameter.

#### `public inline virtual int size() const` {#classdnn__opt_1_1core_1_1solutions_1_1ackley_1a7e6a28edd708637c607069995235874a}

Returns the number of parameters of this solution.

#### Returns
the number of parameters.

#### `public inline virtual vector< float > & get_parameters()` {#classdnn__opt_1_1core_1_1solutions_1_1ackley_1aec209a6af2889a1912400e213301bb1d}

Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.

**See also**: [set_dirty()](#classdnn__opt_1_1core_1_1solution_1a44232576e3a8d1b616f747a24180f397)


#### Returns
a reference to a vector containing all parameters.

#### `public inline virtual `[`ackley`](#classdnn__opt_1_1core_1_1solutions_1_1ackley)` * clone()` {#classdnn__opt_1_1core_1_1solutions_1_1ackley_1a3a54ca55a86dfb6aecd1cfdb5e84baef}

Creates an exact replica of this solution.

#### Returns
a solution instance object that is equal to this solution.

#### `public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` {#classdnn__opt_1_1core_1_1solutions_1_1ackley_1a0b94d6cb8167311e0b47f5e30519a53e}

Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an ackley solution and have the same number of parameters.

#### Parameters
* `a` solution to check if it is assignable to this solution.





#### Returns
true if the given solution is the given solution is assignable, false otherwise.

#### `protected vector< float > _params` {#classdnn__opt_1_1core_1_1solutions_1_1ackley_1a80a2a588e67f96693853969d26a8a24a}



This is a vector containing all the parameters of this solution.

#### `protected inline  ackley(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` {#classdnn__opt_1_1core_1_1solutions_1_1ackley_1a1f395ccfe072000367f90ebbf57631d7}

This is the basic contructor for this class. Is protected since this this class implements the factory pattern. Derived clasess however can use this constructor.

#### Parameters
* `generator` a shared pointer to an instance of a paramter_generator class. The [parameter_generator](#classdnn__opt_1_1core_1_1parameter__generator) is used to populate the parameters of this solution.


* `param_count` is the number of paramters for this solution. Default is 10.

#### `protected inline virtual float calculate_fitness()` {#classdnn__opt_1_1core_1_1solutions_1_1ackley_1a373f76526a0c0553efaba4cf6be2f678}

Calculates the fitness of this solution which in this case is determined by the Ackley function. More information about Ackley function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

#### Returns
the fitness of this solution.

# class `dnn_opt::core::solutions::de_jung` {#classdnn__opt_1_1core_1_1solutions_1_1de__jung}

```
class dnn_opt::core::solutions::de_jung
  : public dnn_opt::core::solution
```  

The [de_jung](#classdnn__opt_1_1core_1_1solutions_1_1de__jung) class represents an optimization solutions which fitness cost is calculated via De' Jung function.

De'Jung function have a global minima in {0,..., 0} with a value of 0. A commonly used search domain for testing is [-5, 5].

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

1.0 

November, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void set(int index,float value)` | Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.
`public inline virtual float get(int index) const` | Returns the value of a given parameter.
`public inline virtual int size() const` | Returns the number of parameters of this solution.
`public inline virtual vector< float > & get_parameters()` | Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.
`public inline virtual `[`de_jung`](#classdnn__opt_1_1core_1_1solutions_1_1de__jung)` * clone()` | Creates an exact replica of this solution.
`public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` | Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an [de_jung](#classdnn__opt_1_1core_1_1solutions_1_1de__jung) solution and have the same number of parameters.
`protected vector< float > _params` | 
`protected inline  de_jung(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` | 
`protected inline virtual float calculate_fitness()` | Calculates the fitness of this solution which in this case is determined by the De'Jung function. More information about De'Jung function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

## Members

#### `public inline virtual void set(int index,float value)` {#classdnn__opt_1_1core_1_1solutions_1_1de__jung_1afb145a3a19084940ed5a7b9072c771e5}

Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.

#### Parameters
* `index` the index of the parameter to be changed. 


* `value` the new parameter's value.

#### `public inline virtual float get(int index) const` {#classdnn__opt_1_1core_1_1solutions_1_1de__jung_1a31dd1912d1fe3cc30c681ed42a0c7461}

Returns the value of a given parameter.

#### Parameters
* `index` the index of the parameter to be retorned.





#### Returns
the current value of the parameter.

#### `public inline virtual int size() const` {#classdnn__opt_1_1core_1_1solutions_1_1de__jung_1a274e438ea89a0d59e322d5afcf5c7e48}

Returns the number of parameters of this solution.

#### Returns
the number of parameters.

#### `public inline virtual vector< float > & get_parameters()` {#classdnn__opt_1_1core_1_1solutions_1_1de__jung_1af6bed9e3bcd06f9c7307b7e231761f3f}

Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.

**See also**: [set_dirty()](#classdnn__opt_1_1core_1_1solution_1a44232576e3a8d1b616f747a24180f397)


#### Returns
a reference to a vector containing all parameters.

#### `public inline virtual `[`de_jung`](#classdnn__opt_1_1core_1_1solutions_1_1de__jung)` * clone()` {#classdnn__opt_1_1core_1_1solutions_1_1de__jung_1a73d103d137af7738ad9f9ed2d22d697b}

Creates an exact replica of this solution.

#### Returns
a solution instance object that is equal to this solution.

#### `public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` {#classdnn__opt_1_1core_1_1solutions_1_1de__jung_1a5ebbcdd5506a09e4ddbfaf5a374d49d1}

Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an [de_jung](#classdnn__opt_1_1core_1_1solutions_1_1de__jung) solution and have the same number of parameters.

#### Parameters
* `a` solution to check if it is assignable to this solution.





#### Returns
true if the given solution is the given solution is assignable, false otherwise.

#### `protected vector< float > _params` {#classdnn__opt_1_1core_1_1solutions_1_1de__jung_1a955d04705744dfa778999775f1ccab00}



This is a vector containing all the parameters of this solution.

#### `protected inline  de_jung(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` {#classdnn__opt_1_1core_1_1solutions_1_1de__jung_1ae95a0edf0d5dcea87a4b7817e3b84a3d}





#### `protected inline virtual float calculate_fitness()` {#classdnn__opt_1_1core_1_1solutions_1_1de__jung_1aefa8bb0336d5d795891a755c5cc7cfc7}

Calculates the fitness of this solution which in this case is determined by the De'Jung function. More information about De'Jung function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

#### Returns
the fitness of this solution.

# class `dnn_opt::core::solutions::griewangk` {#classdnn__opt_1_1core_1_1solutions_1_1griewangk}

```
class dnn_opt::core::solutions::griewangk
  : public dnn_opt::core::solution
```  

The griewangk class.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

1.0 

November, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void set(int index,float value)` | Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.
`public inline virtual float get(int index) const` | Returns the value of a given parameter.
`public inline virtual int size() const` | Returns the number of parameters of this solution.
`public inline virtual vector< float > & get_parameters()` | Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.
`public inline virtual `[`griewangk`](#classdnn__opt_1_1core_1_1solutions_1_1griewangk)` * clone()` | Creates an exact replica of this solution.
`public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` | Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an griewangk solution and have the same number of parameters.
`protected vector< float > _params` | 
`protected inline  griewangk(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` | 
`protected inline virtual float calculate_fitness()` | Calculates the fitness of this solution which in this case is determined by the Griewangk function. More information about Griewangk function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

## Members

#### `public inline virtual void set(int index,float value)` {#classdnn__opt_1_1core_1_1solutions_1_1griewangk_1ae05dbd77111581d3ab647be846de1459}

Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.

#### Parameters
* `index` the index of the parameter to be changed. 


* `value` the new parameter's value.

#### `public inline virtual float get(int index) const` {#classdnn__opt_1_1core_1_1solutions_1_1griewangk_1a2b9ccb0d6d2abdb13403cd74a2d44a13}

Returns the value of a given parameter.

#### Parameters
* `index` the index of the parameter to be retorned.





#### Returns
the current value of the parameter.

#### `public inline virtual int size() const` {#classdnn__opt_1_1core_1_1solutions_1_1griewangk_1a9089e00515643b2235176a4ec4e518b6}

Returns the number of parameters of this solution.

#### Returns
the number of parameters.

#### `public inline virtual vector< float > & get_parameters()` {#classdnn__opt_1_1core_1_1solutions_1_1griewangk_1ab3a7f6d385881056ee13bce15478e695}

Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.

**See also**: [set_dirty()](#classdnn__opt_1_1core_1_1solution_1a44232576e3a8d1b616f747a24180f397)


#### Returns
a reference to a vector containing all parameters.

#### `public inline virtual `[`griewangk`](#classdnn__opt_1_1core_1_1solutions_1_1griewangk)` * clone()` {#classdnn__opt_1_1core_1_1solutions_1_1griewangk_1afd0b5730bcee363c0c6eab5d5bf3a1bf}

Creates an exact replica of this solution.

#### Returns
a solution instance object that is equal to this solution.

#### `public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` {#classdnn__opt_1_1core_1_1solutions_1_1griewangk_1a8ae5517dc2c32055c899c2085df38239}

Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an griewangk solution and have the same number of parameters.

#### Parameters
* `a` solution to check if it is assignable to this solution.





#### Returns
true if the given solution is the given solution is assignable, false otherwise.

#### `protected vector< float > _params` {#classdnn__opt_1_1core_1_1solutions_1_1griewangk_1aa371af63ca653b3733ea811d338917c9}



This is a vector containing all the parameters of this solution.

#### `protected inline  griewangk(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` {#classdnn__opt_1_1core_1_1solutions_1_1griewangk_1aca3459dde1f225b5a8f1e1a62d64aafc}





#### `protected inline virtual float calculate_fitness()` {#classdnn__opt_1_1core_1_1solutions_1_1griewangk_1aaa1a4529fa55549582fe32aaa86d9f27}

Calculates the fitness of this solution which in this case is determined by the Griewangk function. More information about Griewangk function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

#### Returns
the fitness of this solution.

# class `dnn_opt::core::solutions::michalewicz` {#classdnn__opt_1_1core_1_1solutions_1_1michalewicz}

```
class dnn_opt::core::solutions::michalewicz
  : public dnn_opt::core::solution
```  

The michalewicz class.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

1.0 

November, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void set(int index,float value)` | Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.
`public inline virtual float get(int index) const` | Returns the value of a given parameter.
`public inline virtual int size() const` | Returns the number of parameters of this solution.
`public inline virtual vector< float > & get_parameters()` | Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.
`public inline virtual `[`michalewicz`](#classdnn__opt_1_1core_1_1solutions_1_1michalewicz)` * clone()` | Creates an exact replica of this solution.
`public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` | Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an michalewicz solution and have the same number of parameters.
`protected vector< float > _params` | 
`protected inline  michalewicz(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` | 
`protected inline virtual float calculate_fitness()` | Calculates the fitness of this solution which in this case is determined by the Michalewicz function. More information about Michalewicz function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

## Members

#### `public inline virtual void set(int index,float value)` {#classdnn__opt_1_1core_1_1solutions_1_1michalewicz_1a3e1047b13373e9fa8117cdbc465ad4de}

Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.

#### Parameters
* `index` the index of the parameter to be changed. 


* `value` the new parameter's value.

#### `public inline virtual float get(int index) const` {#classdnn__opt_1_1core_1_1solutions_1_1michalewicz_1ada925b43ff4d09ff5dbf9f819f779413}

Returns the value of a given parameter.

#### Parameters
* `index` the index of the parameter to be retorned.





#### Returns
the current value of the parameter.

#### `public inline virtual int size() const` {#classdnn__opt_1_1core_1_1solutions_1_1michalewicz_1a6c8b9217e6bec83b85cdf7c147b8d9b2}

Returns the number of parameters of this solution.

#### Returns
the number of parameters.

#### `public inline virtual vector< float > & get_parameters()` {#classdnn__opt_1_1core_1_1solutions_1_1michalewicz_1a7c9bbb31e93a4df0ff1981a3dedfb8e3}

Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.

**See also**: [set_dirty()](#classdnn__opt_1_1core_1_1solution_1a44232576e3a8d1b616f747a24180f397)


#### Returns
a reference to a vector containing all parameters.

#### `public inline virtual `[`michalewicz`](#classdnn__opt_1_1core_1_1solutions_1_1michalewicz)` * clone()` {#classdnn__opt_1_1core_1_1solutions_1_1michalewicz_1ac102f39f6d688dd433dfcdb375a5588e}

Creates an exact replica of this solution.

#### Returns
a solution instance object that is equal to this solution.

#### `public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` {#classdnn__opt_1_1core_1_1solutions_1_1michalewicz_1ae15281474a71cf1091a1a735136e5e6b}

Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an michalewicz solution and have the same number of parameters.

#### Parameters
* `a` solution to check if it is assignable to this solution.





#### Returns
true if the given solution is the given solution is assignable, false otherwise.

#### `protected vector< float > _params` {#classdnn__opt_1_1core_1_1solutions_1_1michalewicz_1a2bdeeafc8c54d0c71be5ee0189f39043}



This is a vector containing all the parameters of this solution.

#### `protected inline  michalewicz(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` {#classdnn__opt_1_1core_1_1solutions_1_1michalewicz_1a34c3e51c3a1fff8a303d625cf923ab94}





#### `protected inline virtual float calculate_fitness()` {#classdnn__opt_1_1core_1_1solutions_1_1michalewicz_1a86bf5fca4ccb599109ed43f55378cc61}

Calculates the fitness of this solution which in this case is determined by the Michalewicz function. More information about Michalewicz function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

#### Returns
the fitness of this solution.

# class `dnn_opt::core::solutions::network` {#classdnn__opt_1_1core_1_1solutions_1_1network}

```
class dnn_opt::core::solutions::network
  : public dnn_opt::core::solution
```  





## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void set(int,float)` | Changes the value of a given parameter. Derived classes are encouraged call the base class method in order to maintain consistency.
`public inline virtual float get(int index) const` | Returns the value of a given parameter.
`public inline virtual int size() const` | Returns the number of parameters of this solution.
`public inline virtual vector< float > & get_parameters()` | `get_parameters` returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified calling
`public inline bool is_lazzy()` | Returns if this solution calculates its fitness in a lazzy way.
`public inline void set_lazzy(bool lazzy)` | Changes the way this solution calculates its fitness.
`public inline virtual `[`network`](#classdnn__opt_1_1core_1_1solutions_1_1network)` * clone()` | Creates an exact replica of this solution. The procedure to create a replica of a solution depends on the nature of the solution and must be defined in derived classes.
`public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` | Determines if the given object instance is assignable to this solution. This decision depends on the nature of the solution and must be defined in derived classes. Usually this comprobation must check if the given solution is of the same type, the same number of parameters.
`public inline void add_layer(shared_ptr< `[`layer`](#classdnn__opt_1_1core_1_1layer)` > l)` | 
`public inline `[`network`](#classdnn__opt_1_1core_1_1solutions_1_1network)` & operator<<(shared_ptr< `[`layer`](#classdnn__opt_1_1core_1_1layer)` > l)` | 
`protected vector< float > _params` | 
`protected vector< shared_ptr< `[`layer`](#classdnn__opt_1_1core_1_1layer)` > > _layers` | 
`protected shared_ptr< `[`sampler`](#classdnn__opt_1_1core_1_1sampler)` > _s` | 
`protected shared_ptr< `[`error_function`](#classdnn__opt_1_1core_1_1error__function)` > _E` | 
`protected bool _lazzy` | 
`protected inline  network(bool lazzy,shared_ptr< `[`sampler`](#classdnn__opt_1_1core_1_1sampler)` > s,shared_ptr< `[`error_function`](#classdnn__opt_1_1core_1_1error__function)` > E,shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` | 
`protected inline virtual float calculate_fitness()` | Calculates the fitness of this solution. The nature of the fitness depends on the nature of the solution and must be implemented in derived classes.
`protected inline virtual float error(const vector< vector< float > > & input,const vector< vector< float > > & expected)` | 
`protected inline vector< float > predict(const vector< float > & in)` | 

## Members

#### `public inline virtual void set(int,float)` {#classdnn__opt_1_1core_1_1solutions_1_1network_1ae36c56d56a00aa2964b85f1b38d3f017}

Changes the value of a given parameter. Derived classes are encouraged call the base class method in order to maintain consistency.

#### Parameters
* `index` the index of the parameter to be changed. 


* `value` the new parameter's value.

#### `public inline virtual float get(int index) const` {#classdnn__opt_1_1core_1_1solutions_1_1network_1a7d0f95a4ffb7954da581095e2889e7d8}

Returns the value of a given parameter.

#### Parameters
* `index` the index of the parameter to be returned.





#### Returns
the current value of the parameter.

#### `public inline virtual int size() const` {#classdnn__opt_1_1core_1_1solutions_1_1network_1a47cf18b1421b1a656dca1a778ba2d6a3}

Returns the number of parameters of this solution.

#### Returns
the number of parameters.

#### `public inline virtual vector< float > & get_parameters()` {#classdnn__opt_1_1core_1_1solutions_1_1network_1a686417ffc09a2a96eb700bc126790c7a}

`get_parameters` returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified calling

**See also**: [set_dirty()](#classdnn__opt_1_1core_1_1solution_1a44232576e3a8d1b616f747a24180f397)


#### Returns
a reference to a vector containing all parameters.

#### `public inline bool is_lazzy()` {#classdnn__opt_1_1core_1_1solutions_1_1network_1a99eef86f2d7c5c883ecafd4ab3cb6ab2}

Returns if this solution calculates its fitness in a lazzy way.

#### Returns
true if fitness is lazzy, false otherwise.

#### `public inline void set_lazzy(bool lazzy)` {#classdnn__opt_1_1core_1_1solutions_1_1network_1aeb38e87cc035a089fcf77f51252d0f52}

Changes the way this solution calculates its fitness.

#### Parameters
* `lazzy` set true for lazzy fitness false for not lazzy.

#### `public inline virtual `[`network`](#classdnn__opt_1_1core_1_1solutions_1_1network)` * clone()` {#classdnn__opt_1_1core_1_1solutions_1_1network_1a94fa7ae58b1862ab7a1f49cbd02f63f1}

Creates an exact replica of this solution. The procedure to create a replica of a solution depends on the nature of the solution and must be defined in derived classes.

#### Returns
a solution instance that is equal to this solution.

#### `public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` {#classdnn__opt_1_1core_1_1solutions_1_1network_1ac1cdf08e2ecedfa7ad7d42f37836138a}

Determines if the given object instance is assignable to this solution. This decision depends on the nature of the solution and must be defined in derived classes. Usually this comprobation must check if the given solution is of the same type, the same number of parameters.

#### Parameters
* `a` solution to check if it is assignable to this solution.





#### Returns
true if the given solution is the given solution is assignable, false otherwise.

#### `public inline void add_layer(shared_ptr< `[`layer`](#classdnn__opt_1_1core_1_1layer)` > l)` {#classdnn__opt_1_1core_1_1solutions_1_1network_1a11d39fcf6ca0cd69ce0bef17106bf439}





#### `public inline `[`network`](#classdnn__opt_1_1core_1_1solutions_1_1network)` & operator<<(shared_ptr< `[`layer`](#classdnn__opt_1_1core_1_1layer)` > l)` {#classdnn__opt_1_1core_1_1solutions_1_1network_1a2536cd3dc40d4e8d5a66c1af27b5b029}





#### `protected vector< float > _params` {#classdnn__opt_1_1core_1_1solutions_1_1network_1aefd6c535e8328891f188268b6e3a5f42}





#### `protected vector< shared_ptr< `[`layer`](#classdnn__opt_1_1core_1_1layer)` > > _layers` {#classdnn__opt_1_1core_1_1solutions_1_1network_1a85e04597c640d833fb72cf7ebcd97aa1}





#### `protected shared_ptr< `[`sampler`](#classdnn__opt_1_1core_1_1sampler)` > _s` {#classdnn__opt_1_1core_1_1solutions_1_1network_1aca91f68e659f217767e29739544be5d2}





#### `protected shared_ptr< `[`error_function`](#classdnn__opt_1_1core_1_1error__function)` > _E` {#classdnn__opt_1_1core_1_1solutions_1_1network_1a804f9eba3d00778e9a0e49fe12b171b1}





#### `protected bool _lazzy` {#classdnn__opt_1_1core_1_1solutions_1_1network_1aa50a61b28e6782cf1b75d45793f76e46}





#### `protected inline  network(bool lazzy,shared_ptr< `[`sampler`](#classdnn__opt_1_1core_1_1sampler)` > s,shared_ptr< `[`error_function`](#classdnn__opt_1_1core_1_1error__function)` > E,shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` {#classdnn__opt_1_1core_1_1solutions_1_1network_1ae616b6b154d06acbb7ed97a14f757e1a}





#### `protected inline virtual float calculate_fitness()` {#classdnn__opt_1_1core_1_1solutions_1_1network_1adc1bef1f5f1713cfc732bd66fcd25169}

Calculates the fitness of this solution. The nature of the fitness depends on the nature of the solution and must be implemented in derived classes.

#### Returns
the fitness of this solution.

#### `protected inline virtual float error(const vector< vector< float > > & input,const vector< vector< float > > & expected)` {#classdnn__opt_1_1core_1_1solutions_1_1network_1a6b092516e3a2d4965536f648f29ae7a3}





#### `protected inline vector< float > predict(const vector< float > & in)` {#classdnn__opt_1_1core_1_1solutions_1_1network_1a0cc2eb0a7274379f7bef14710602ee64}





# class `dnn_opt::core::solutions::rastrigin` {#classdnn__opt_1_1core_1_1solutions_1_1rastrigin}

```
class dnn_opt::core::solutions::rastrigin
  : public dnn_opt::core::solution
```  

The rastrigin class represents an optimization solutions which fitness cost is calculated via Rastrigin function.

Rastrigin function have a global minima in {0,..., 0} with a value of 0. A commonly used search domain for testing is [-5.12, 5.12].

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

1.0 

November, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void set(int index,float value)` | Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.
`public inline virtual float get(int index) const` | Returns the value of a given parameter.
`public inline virtual int size() const` | Returns the number of parameters of this solution.
`public inline virtual vector< float > & get_parameters()` | Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.
`public inline virtual `[`rastrigin`](#classdnn__opt_1_1core_1_1solutions_1_1rastrigin)` * clone()` | Creates an exact replica of this solution.
`public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` | Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an rastrigin solution and have the same number of parameters.
`protected vector< float > _params` | 
`protected inline  rastrigin(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` | 
`protected inline virtual float calculate_fitness()` | Calculates the fitness of this solution which in this case is determined by the Rastrigin function. More information about Rastrigin function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

## Members

#### `public inline virtual void set(int index,float value)` {#classdnn__opt_1_1core_1_1solutions_1_1rastrigin_1abd4af52cdf280b488639a0c5ffbdaf88}

Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.

#### Parameters
* `index` the index of the parameter to be changed. 


* `value` the new parameter's value.

#### `public inline virtual float get(int index) const` {#classdnn__opt_1_1core_1_1solutions_1_1rastrigin_1a3d22e8ce485f00030c6c876018eb8949}

Returns the value of a given parameter.

#### Parameters
* `index` the index of the parameter to be retorned.





#### Returns
the current value of the parameter.

#### `public inline virtual int size() const` {#classdnn__opt_1_1core_1_1solutions_1_1rastrigin_1a0c43d4517420cf56618758e23ac3b243}

Returns the number of parameters of this solution.

#### Returns
the number of parameters.

#### `public inline virtual vector< float > & get_parameters()` {#classdnn__opt_1_1core_1_1solutions_1_1rastrigin_1a1e3e2fd075748c381b4bb8ee0d7fac1b}

Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.

**See also**: [set_dirty()](#classdnn__opt_1_1core_1_1solution_1a44232576e3a8d1b616f747a24180f397)


#### Returns
a reference to a vector containing all parameters.

#### `public inline virtual `[`rastrigin`](#classdnn__opt_1_1core_1_1solutions_1_1rastrigin)` * clone()` {#classdnn__opt_1_1core_1_1solutions_1_1rastrigin_1ab6f1dbdb3fb77c8875ffd481bb1eff27}

Creates an exact replica of this solution.

#### Returns
a solution instance object that is equal to this solution.

#### `public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` {#classdnn__opt_1_1core_1_1solutions_1_1rastrigin_1a0256045b92e3b1931c9e6e8637d94dca}

Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an rastrigin solution and have the same number of parameters.

#### Parameters
* `a` solution to check if it is assignable to this solution.





#### Returns
true if the given solution is the given solution is assignable, false otherwise.

#### `protected vector< float > _params` {#classdnn__opt_1_1core_1_1solutions_1_1rastrigin_1ac55568de677461978bb572c4ece9f151}



This is a vector containing all the parameters of this solution.

#### `protected inline  rastrigin(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` {#classdnn__opt_1_1core_1_1solutions_1_1rastrigin_1a2244c9704e3f94091ad50f72474ca6a5}





#### `protected inline virtual float calculate_fitness()` {#classdnn__opt_1_1core_1_1solutions_1_1rastrigin_1ac65dac31c42d4740edd9464dc0beb2de}

Calculates the fitness of this solution which in this case is determined by the Rastrigin function. More information about Rastrigin function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

#### Returns
the fitness of this solution.

# class `dnn_opt::core::solutions::rosenbrock` {#classdnn__opt_1_1core_1_1solutions_1_1rosenbrock}

```
class dnn_opt::core::solutions::rosenbrock
  : public dnn_opt::core::solution
```  

The rosenbrock class represents an optimization solutions which fitness cost is calculated via Rosenbrock function.

Rosenbrock function have a global minima in {1,..., 1} with a value of 0. A commonly used search domain for testing is [-5, 5].

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

1.0 

November, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void set(int index,float value)` | Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.
`public inline virtual float get(int index) const` | Returns the value of a given parameter.
`public inline virtual int size() const` | Returns the number of parameters of this solution.
`public inline virtual vector< float > & get_parameters()` | Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.
`public inline virtual `[`rosenbrock`](#classdnn__opt_1_1core_1_1solutions_1_1rosenbrock)` * clone()` | Creates an exact replica of this solution.
`public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` | Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an rosenbrock solution and have the same number of parameters.
`protected vector< float > _params` | 
`protected inline  rosenbrock(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` | 
`protected inline virtual float calculate_fitness()` | Calculates the fitness of this solution which in this case is determined by the Rosenbrock function. More information about Rosenbrock function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

## Members

#### `public inline virtual void set(int index,float value)` {#classdnn__opt_1_1core_1_1solutions_1_1rosenbrock_1a0852c4e849071f0f0c06e0754e5ac7d3}

Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.

#### Parameters
* `index` the index of the parameter to be changed. 


* `value` the new parameter's value.

#### `public inline virtual float get(int index) const` {#classdnn__opt_1_1core_1_1solutions_1_1rosenbrock_1a861588643f4a5da20c8f01da9ffb85c4}

Returns the value of a given parameter.

#### Parameters
* `index` the index of the parameter to be retorned.





#### Returns
the current value of the parameter.

#### `public inline virtual int size() const` {#classdnn__opt_1_1core_1_1solutions_1_1rosenbrock_1a7125978b48d43fe5db7f3493fc0ceef3}

Returns the number of parameters of this solution.

#### Returns
the number of parameters.

#### `public inline virtual vector< float > & get_parameters()` {#classdnn__opt_1_1core_1_1solutions_1_1rosenbrock_1a0304ff92d39b3a9e743f46bde5b3a96b}

Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.

**See also**: [set_dirty()](#classdnn__opt_1_1core_1_1solution_1a44232576e3a8d1b616f747a24180f397)


#### Returns
a reference to a vector containing all parameters.

#### `public inline virtual `[`rosenbrock`](#classdnn__opt_1_1core_1_1solutions_1_1rosenbrock)` * clone()` {#classdnn__opt_1_1core_1_1solutions_1_1rosenbrock_1ac549903e0bf7af430b691c4156dff62f}

Creates an exact replica of this solution.

#### Returns
a solution instance object that is equal to this solution.

#### `public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` {#classdnn__opt_1_1core_1_1solutions_1_1rosenbrock_1a0eccc3a47d7e30a196108e278004ccdd}

Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an rosenbrock solution and have the same number of parameters.

#### Parameters
* `a` solution to check if it is assignable to this solution.





#### Returns
true if the given solution is the given solution is assignable, false otherwise.

#### `protected vector< float > _params` {#classdnn__opt_1_1core_1_1solutions_1_1rosenbrock_1ad88182faba3c1a32aa88fb3a2238ca43}



This is a vector containing all the parameters of this solution.

#### `protected inline  rosenbrock(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` {#classdnn__opt_1_1core_1_1solutions_1_1rosenbrock_1a85893d1aacc66a3037e7980cca9adcf0}





#### `protected inline virtual float calculate_fitness()` {#classdnn__opt_1_1core_1_1solutions_1_1rosenbrock_1ad004d213f4ece773554fa298e8c35031}

Calculates the fitness of this solution which in this case is determined by the Rosenbrock function. More information about Rosenbrock function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

#### Returns
the fitness of this solution.

# class `dnn_opt::core::solutions::schwefel` {#classdnn__opt_1_1core_1_1solutions_1_1schwefel}

```
class dnn_opt::core::solutions::schwefel
  : public dnn_opt::core::solution
```  

The schwefel class.

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

1.0 

November, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void set(int index,float value)` | Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.
`public inline virtual float get(int index) const` | Returns the value of a given parameter.
`public inline virtual int size() const` | Returns the number of parameters of this solution.
`public inline virtual vector< float > & get_parameters()` | Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.
`public inline virtual `[`schwefel`](#classdnn__opt_1_1core_1_1solutions_1_1schwefel)` * clone()` | Creates an exact replica of this solution.
`public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` | Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an schwefel solution and have the same number of parameters.
`protected vector< float > _params` | 
`protected inline  schwefel(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` | 
`protected inline virtual float calculate_fitness()` | Calculates the fitness of this solution which in this case is determined by the Schwefel function. More information about Schwefel function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

## Members

#### `public inline virtual void set(int index,float value)` {#classdnn__opt_1_1core_1_1solutions_1_1schwefel_1a126572a313393445250ffc3da3a74840}

Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.

#### Parameters
* `index` the index of the parameter to be changed. 


* `value` the new parameter's value.

#### `public inline virtual float get(int index) const` {#classdnn__opt_1_1core_1_1solutions_1_1schwefel_1afa94e7ad99513a72b4b37d9ad8337bbe}

Returns the value of a given parameter.

#### Parameters
* `index` the index of the parameter to be retorned.





#### Returns
the current value of the parameter.

#### `public inline virtual int size() const` {#classdnn__opt_1_1core_1_1solutions_1_1schwefel_1a2f4bb2134b580cc6b075e1b5c0b4bd5a}

Returns the number of parameters of this solution.

#### Returns
the number of parameters.

#### `public inline virtual vector< float > & get_parameters()` {#classdnn__opt_1_1core_1_1solutions_1_1schwefel_1afc0e77a075cbe7d495a739e279ba390b}

Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.

**See also**: [set_dirty()](#classdnn__opt_1_1core_1_1solution_1a44232576e3a8d1b616f747a24180f397)


#### Returns
a reference to a vector containing all parameters.

#### `public inline virtual `[`schwefel`](#classdnn__opt_1_1core_1_1solutions_1_1schwefel)` * clone()` {#classdnn__opt_1_1core_1_1solutions_1_1schwefel_1ab02d2cf95c870a06fedd57f944c79e02}

Creates an exact replica of this solution.

#### Returns
a solution instance object that is equal to this solution.

#### `public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` {#classdnn__opt_1_1core_1_1solutions_1_1schwefel_1ade2b83f5abf689867a44b9558f17b8bd}

Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an schwefel solution and have the same number of parameters.

#### Parameters
* `a` solution to check if it is assignable to this solution.





#### Returns
true if the given solution is the given solution is assignable, false otherwise.

#### `protected vector< float > _params` {#classdnn__opt_1_1core_1_1solutions_1_1schwefel_1acb310ad7f2beaf7c171c549c35f7dce2}



This is a vector containing all the parameters of this solution.

#### `protected inline  schwefel(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` {#classdnn__opt_1_1core_1_1solutions_1_1schwefel_1aad87e4ae0d1670d7d1b9c5fe55ab679a}





#### `protected inline virtual float calculate_fitness()` {#classdnn__opt_1_1core_1_1solutions_1_1schwefel_1a370a5df9503dde06d12a9824b7f4a4e8}

Calculates the fitness of this solution which in this case is determined by the Schwefel function. More information about Schwefel function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

#### Returns
the fitness of this solution.

# class `dnn_opt::core::solutions::styblinski_tang` {#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang}

```
class dnn_opt::core::solutions::styblinski_tang
  : public dnn_opt::core::solution
```  

The [styblinski_tang](#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang) class represents an optimization solutions which fitness cost is calculated via Styblinski-Tang function.

Styblinski-Tang function have a global minima in {-2.093,..., 2.9053} with a value of -39.16. A commonly used search domain for testing is [-5, 5].

Jairo Rojas-Delgado [jrdelgado@uci.cu](mailto:jrdelgado@uci.cu)

1.0 

November, 2016

## Summary

 Members                        | Descriptions                                
--------------------------------|---------------------------------------------
`public inline virtual void set(int index,float value)` | Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.
`public inline virtual float get(int index) const` | Returns the value of a given parameter.
`public inline virtual int size() const` | Returns the number of parameters of this solution.
`public inline virtual vector< float > & get_parameters()` | Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.
`public inline virtual `[`styblinski_tang`](#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang)` * clone()` | Creates an exact replica of this solution.
`public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` | Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an [styblinski_tang](#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang) solution and have the same number of parameters.
`protected vector< float > _params` | 
`protected inline  styblinski_tang(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` | 
`protected inline virtual float calculate_fitness()` | Calculates the fitness of this solution which in this case is determined by the Styblinski-Tang function. More information about Styblinski-Tang function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

## Members

#### `public inline virtual void set(int index,float value)` {#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang_1a35a3bad4b9e4997989d3341443956b79}

Changes the value of a given parameter. Derived classes must allways call the base class method in order to maintain consistency.

#### Parameters
* `index` the index of the parameter to be changed. 


* `value` the new parameter's value.

#### `public inline virtual float get(int index) const` {#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang_1a7b1423ec4ce7a9e7dcf14bda92346d2b}

Returns the value of a given parameter.

#### Parameters
* `index` the index of the parameter to be retorned.





#### Returns
the current value of the parameter.

#### `public inline virtual int size() const` {#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang_1ae6af24b13389d39b0fd29f3d57b8e7b0}

Returns the number of parameters of this solution.

#### Returns
the number of parameters.

#### `public inline virtual vector< float > & get_parameters()` {#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang_1a3924262412a1f62b0f13775928b4efe0}

Returns a reference to a vector containing all parameters of the solution. If parameters are modified via this vector the solution should be notified called.

**See also**: [set_dirty()](#classdnn__opt_1_1core_1_1solution_1a44232576e3a8d1b616f747a24180f397)


#### Returns
a reference to a vector containing all parameters.

#### `public inline virtual `[`styblinski_tang`](#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang)` * clone()` {#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang_1a2b33027209c5fdf39818c8211a39271e}

Creates an exact replica of this solution.

#### Returns
a solution instance object that is equal to this solution.

#### `public inline virtual bool assignable(`[`solution`](#classdnn__opt_1_1core_1_1solution)` & s) const` {#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang_1ab44d726e43a08e358079b73cba9bf53e}

Determines if the given object instance is assignable to this solution. A solution if assignable to this one if is an [styblinski_tang](#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang) solution and have the same number of parameters.

#### Parameters
* `a` solution to check if it is assignable to this solution.





#### Returns
true if the given solution is the given solution is assignable, false otherwise.

#### `protected vector< float > _params` {#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang_1aa427218b5c1aba2555b26fe5af7dd343}



This is a vector containing all the parameters of this solution.

#### `protected inline  styblinski_tang(shared_ptr< `[`parameter_generator`](#classdnn__opt_1_1core_1_1parameter__generator)` > generator,int param_count)` {#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang_1a36cde749c21dfd98e1682fbe9fb67a27}





#### `protected inline virtual float calculate_fitness()` {#classdnn__opt_1_1core_1_1solutions_1_1styblinski__tang_1a5f51d99bdb0e7aa8764ebba6587eea48}

Calculates the fitness of this solution which in this case is determined by the Styblinski-Tang function. More information about Styblinski-Tang function can be found in [Wikipedia](https://en.wikipedia.org/wiki/Test_functions_for_optimization).

#### Returns
the fitness of this solution.

