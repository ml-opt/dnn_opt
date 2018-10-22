#ifdef ENABLE_CORE

  /* base */
  #include <core/base/activation.h>
  #include <core/base/algorithm.h>
  #include <core/base/error.h>
  #include <core/base/generator.h>
  #include <core/base/layer.h>
  #include <core/base/reader.h>
  #include <core/base/sampler.h>
  #include <core/base/solution.h>
  #include <core/base/solution_set.h>
  #include <core/base/shufler.h>

  /* algorithms */
  #include <core/algorithms/pso.h>
  #include <core/algorithms/firefly.h>
  #include <core/algorithms/cuckoo.h>
  #include <core/algorithms/continuation.h>
  #include <core/algorithms/opwa.h>

  /* errors */
  #include <core/errors/mse.h>
  #include <core/errors/overall.h>

  /* generators */
  #include <core/generators/constant.h>
  #include <core/generators/normal.h>
  #include <core/generators/uniform.h>

  /* solutions */
  #include <core/solutions/network.h>
  #include <core/solutions/hyper.h>

  #include <core/solutions/ackley.h>
  #include <core/solutions/de_jung.h>
  #include <core/solutions/griewangk.h>
  #include <core/solutions/michalewicz.h>
  #include <core/solutions/rastrigin.h>
  #include <core/solutions/rosenbrock.h>
  #include <core/solutions/schwefel.h>
  #include <core/solutions/styblinski_tang.h>
  #include <core/solutions/hyper.h>

  /* layers */
  #include <core/layers/fc.h>

  /* activations */
  #include <core/activations/elu.h>
  #include <core/activations/hard_limit.h>
  #include <core/activations/identity.h>
  #include <core/activations/relu.h>
  #include <core/activations/sigmoid.h>
  #include <core/activations/tan_h.h>

  /* readers */
  #include <core/readers/file_reader.h>

#endif

#ifdef ENABLE_CUDA

  /* base */
  #include <cuda/base/activation.h>
  #include <cuda/base/algorithm.h>
  #include <cuda/base/error.h>
  #include <cuda/base/generator.h>
  #include <cuda/base/layer.h>
  #include <cuda/base/reader.h>
  #include <cuda/base/sampler.h>
  #include <cuda/base/solution.h>
  #include <cuda/base/solution_set.h>

  /* algorithms */
//  #include <cuda/algorithms/pso.h>
  #include <cuda/algorithms/firefly.h>
  #include <cuda/algorithms/continuation.h>
  #include <cuda/algorithms/opwa.h>

  /* errors */
  #include <cuda/errors/mse.h>
//  #include <cuda/errors/overall.h>

  /* generators */
//  #include <cuda/generators/constant.h>
//  #include <cuda/generators/normal.h>
  #include <cuda/generators/uniform.h>

  /* solutions */
  #include <cuda/solutions/network.h>
//  #include <cuda/solutions/hyper.h>

//  #include <cuda/solutions/ackley.h>
  #include <cuda/solutions/de_jung.h>
//  #include <cuda/solutions/griewangk.h>
//  #include <cuda/solutions/michalewicz.h>
//  #include <cuda/solutions/rastrigin.h>
//  #include <cuda/solutions/rosenbrock.h>
//  #include <cuda/solutions/schwefel.h>
//  #include <cuda/solutions/styblinski_tang.h>

  /* layers */
  #include <cuda/layers/fc.h>

  /* activations */
  #include <cuda/activations/elu.h>
  #include <cuda/activations/hard_limit.h>
  #include <cuda/activations/identity.h>
  #include <cuda/activations/relu.h>
  #include <cuda/activations/sigmoid.h>
  #include <cuda/activations/tan_h.h>

  /* readers */
  #include <cuda/readers/file_reader.h>

#endif
