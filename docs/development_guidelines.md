# Development guidelines

This section contains general development guidelines for the dnn_opt library. Plase read carefully the following items to grasp a general idea of the structure and function of the library before contributing.

## Library structure

The library is designed to consider several execution plattforms (e.g. CPU, GPU, etc.) in a scalable and efficient way. All function in the library should be implemented inside the dnn_opt namespace. Additionally, the library defines a separated namespace within dnn_opt for each execution plattform. This way, currently there are four nested namespaces inside dnn_opt:

1. dnn_opt::core that contains sequential implementations of library functionalities. Under this namespace, you can find all library funcionalities and there is not depedency with third-party libraries or software. Even when eficiency is important, the implementation in this namespace is intended to be more educational, hence obscure code should be avoided.

2. dnn_opt::copt is built on top on dnn_opt::core namespace. Functions in this namespace are the same than in dnn_opt::core, however, whenever possible manually coded functions that have better implementations in third-party software, such as matrix multiplications in BLAS, are replaced.

3. dnn_opt::mcpu is built on top on dnn_opt::copt. Function in this namespace are the same than in dnn_opt::core, however, implementation is adapted to exploit multiple-core CPU hardware by means of using specific third-party libraries such as OpenMP.

4. dnn_opt::cuda is built on top of dnn_opt::core. Function in this namespace are the same than in dnn_opt::core, however, implementation is adapted to exploit Nvidia GPU hardware by means of using specific third-party libraries such as Thrust.

As you can see, in each namespace the same functions are provided. For this reason, all clasess headers are quite similar in each namespace, only implementation is changed. The rule of thumb when extending the library with new functionalities is as follows:

1. Create the basic implementation in the dnn_opt::core namespace.

2. Create the new class in the target namespace. Make sure that the new class extends from the same classes in the target namespaces than the original class in the dnn_opt::core namespace. Additionally, the new class should inherit from the original class in the dnn_opt::core namespace.

You need to consider specific details of each namespace, for example in dnn_opt::cuda data is stored in GPU memory space instead of RAM. Notice that the public interface is the same in each namespace. This will ensure a fundamental property of the library: the use of the library is the same regarding the underliying implementation and target plattform. Read from existing documentation. Take a look to the examples.

## Contributing

Plase follow the next steeps to contribute to the library:

1. Consider what new functionalty you want to provide.
2. Open an issue explaining why you want to implement the new function and how you think to do it. Ask for feed back about the best way to do it to the community.
3. Fork from the project and implement the new function. Make sure to follow the code standards for the project.
4. Create a Pull Request referencing the opened issue. Ask for review of your code. Be ready to explain your approach and reply to issues. 
