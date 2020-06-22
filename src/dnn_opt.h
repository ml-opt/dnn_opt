/*************************** CORE INCLUDES ************************************/

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
  #include <core/base/set.h>
  #include <core/base/shufler.h>
  #include <core/base/proxy_sampler.h>

  /* algorithms */
  #include <core/algorithms/pso.h>
  #include <core/algorithms/firefly.h>
  #include <core/algorithms/cuckoo.h>
  #include <core/algorithms/continuation.h>
  #include <core/algorithms/opwa.h>
  #include <core/algorithms/early_stop.h>
  #include <core/algorithms/gwo.h>

  /* errors */
  #include <core/errors/mse.h>
  #include <core/errors/overall.h>

  /* generators */
  #include <core/generators/constant.h>
  #include <core/generators/normal.h>
  #include <core/generators/uniform.h>

  /* solutions */
  #include <core/solutions/network.h>
  #include <core/solutions/bench/step.h>
  #include <core/solutions/bench/brown_function.h>
  #include <core/solutions/bench/cosine_mixture.h>
  #include <core/solutions/bench/chung_reynolds.h>
  #include <core/solutions/bench/csendes.h>
  #include <core/solutions/bench/deb1.h>
  #include <core/solutions/bench/deb3.h>
  #include <core/solutions/bench/dixonp.h>
  #include <core/solutions/bench/eggh.h>
  #include <core/solutions/bench/expo.h>
  #include <core/solutions/bench/giunta.h>
  #include <core/solutions/bench/alpine.h>
  #include <core/solutions/bench/ackley.h>
  #include <core/solutions/bench/de_jung.h>
  #include <core/solutions/bench/griewangk.h>
  #include <core/solutions/bench/rastrigin.h>
  #include <core/solutions/bench/rosenbrock.h>
  #include <core/solutions/bench/schwefel.h>
  #include <core/solutions/bench/styblinski_tang.h>
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
  #include <core/readers/csv_reader.h>

  /* statics */
  #include <core/statics/cv.h>

#endif

/*************************** COPT INCLUDES ************************************/

#ifdef ENABLE_COPT

  #ifndef ENABLE_CORE
  static_assert(false, "dnn_opt::copt requires dnn_opt::core wrapper");
  #endif

  /* base */
  #include <copt/base/activation.h>
  #include <copt/base/algorithm.h>
  #include <copt/base/error.h>
  #include <copt/base/generator.h>
  #include <copt/base/layer.h>
  #include <copt/base/reader.h>
  #include <copt/base/sampler.h>
  #include <copt/base/solution.h>
  #include <copt/base/set.h>
  #include <copt/base/shufler.h>
  #include <copt/base/proxy_sampler.h>

  /* algorithms */
  #include <copt/algorithms/pso.h>
  #include <copt/algorithms/firefly.h>
  #include <copt/algorithms/cuckoo.h>
  #include <copt/algorithms/continuation.h>
  #include <copt/algorithms/opwa.h>
  #include <copt/algorithms/early_stop.h>

  /* errors */
  #include <copt/errors/mse.h>
  #include <copt/errors/overall.h>

  /* generators */
  #include <copt/generators/constant.h>
  #include <copt/generators/normal.h>
  #include <copt/generators/uniform.h>

  /* solutions */
  #include <copt/solutions/network.h>
  #include <copt/solutions/step.h>
  #include <copt/solutions/alpine.h>
  #include <copt/solutions/ackley.h>
  #include <copt/solutions/de_jung.h>
  #include <copt/solutions/griewangk.h>
  #include <copt/solutions/rastrigin.h>
  #include <copt/solutions/rosenbrock.h>
  #include <copt/solutions/schwefel.h>
  #include <copt/solutions/styblinski_tang.h>
  #include <copt/solutions/hyper.h>

  /* layers */
  #include <copt/layers/fc.h>

  /* activations */
  #include <copt/activations/elu.h>
  #include <copt/activations/hard_limit.h>
  #include <copt/activations/identity.h>
  #include <copt/activations/relu.h>
  #include <copt/activations/sigmoid.h>
  #include <copt/activations/tan_h.h>

  /* readers */
  #include <copt/readers/csv_reader.h>

  /* statics */
  #include <copt/statics/cv.h>

#endif

/*************************** CUDA INCLUDES ************************************/

#ifdef ENABLE_CUDA

  #ifndef ENABLE_CORE
  static_assert(false, "dnn_opt::cuda requires dnn_opt::copt wrapper");
  #endif

  /* base */
  #include <cuda/base/activation.h>
  #include <cuda/base/algorithm.h>
  #include <cuda/base/error.h>
  #include <cuda/base/generator.h>
  #include <cuda/base/layer.h>
  #include <cuda/base/reader.h>
  #include <cuda/base/sampler.h>
  #include <cuda/base/solution.h>
  #include <cuda/base/set.h>
  #include <cuda/base/proxy_sampler.h>

  /* algorithms */
  #include <cuda/algorithms/pso.h>
  #include <cuda/algorithms/firefly.h>
  #include <cuda/algorithms/cuckoo.h>
  #include <cuda/algorithms/continuation.h>
  #include <cuda/algorithms/opwa.h>

  /* errors */
  #include <cuda/errors/mse.h>
//  #include <cuda/errors/overall.h>

  /* generators */
  #include <cuda/generators/constant.h>
  #include <cuda/generators/normal.h>
  #include <cuda/generators/uniform.h>

  /* solutions */
  #include <cuda/solutions/network.h>
  #include <cuda/solutions/step.h>
  #include <cuda/solutions/alpine.h>
  #include <cuda/solutions/ackley.h>
  #include <cuda/solutions/de_jung.h>
  #include <cuda/solutions/griewangk.h>
  #include <cuda/solutions/rastrigin.h>
  #include <cuda/solutions/rosenbrock.h>
  #include <cuda/solutions/schwefel.h>
  #include <cuda/solutions/styblinski_tang.h>
  #include <cuda/solutions/hyper.h>

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
  #include <cuda/readers/csv_reader.h>

#endif
