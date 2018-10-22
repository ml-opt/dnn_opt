#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <cuda/layers/fc.h>

namespace dnn_opt
{
namespace cuda
{
namespace layers
{

namespace ops
{
namespace fc
{

/**
 * @brief Thrust functor to transform an input sequence into indexes.
 *
 * @author Jairo Rojas-Delgado<jrdelgado@uci.cu>
 * @date September, 2017
 * @version 1.0
 */
struct in_seq_op : public thrust::unary_function<int, int>
{
public:

  in_seq_op(int range, int repeat) : _range(range), _repeat(repeat)
  {

  }
  __host__ __device__
  int operator()(const int &value) const
  {
    return (value % _range) + _range * (value / (_range * _repeat));
  }

private:

  int _range;
  int _repeat;

};

/**
 * @brief Thrust functor to transform an input sequence into indexes.
 *
 * @author Jairo Rojas-Delgado<jrdelgado@uci.cu>
 * @date September, 2017
 * @version 1.0
 */
struct bias_op : public thrust::unary_function<int, int>
{
public:

  bias_op(int start, int size) : _start(start), _size(size)
  {

  }
  __host__ __device__
  int operator()(const int &value) const
  {
    return _start + (value % _size);
  }

private:

  int _start;
  int _size;

};

/**
 * @brief Thrust functor to perform an unary division rest of a sequence by a
 * constant number.
 *
 * @author Jairo Rojas-Delgado<jrdelgado@uci.cu>
 * @date September, 2017
 * @version 1.0
 */
struct mod_op : public thrust::unary_function<int, int>
{
public:

  mod_op(int a) : _a(a)
  {

  }

  __host__ __device__
  int operator()(const int &value) const
  {
    return value % _a;
  }

private:

  int _a;

};

/**
 * @brief Thrust functor to perform an unary division of a sequence by a
 * constant number.
 *
 * @author Jairo Rojas-Delgado<jrdelgado@uci.cu>
 * @date September, 2017
 * @version 1.0
 */
struct div_op : public thrust::unary_function<int, int>
{
public:

  div_op(int a) : _a(a)
  {

  }

  __host__ __device__
  int operator()(const int &value) const
  {
    return value / _a;
  }

private:

  int _a;

};

/**
 * @brief Thrust functor to perform multiplication operation of over a zip
 * sequence.
 *
 * @author Jairo Rojas-Delgado<jrdelgado@uci.cu>
 * @date September, 2017
 * @version 1.0
 */
struct mult : public thrust::unary_function<float, float>
{
public:

  template <typename Tuple>
  __host__ __device__
  float operator()(Tuple t) const
  {
    float a = thrust::get<0>(t);
    float b = thrust::get<1>(t);

    return a * b;
  }
};

} // namespace fc
} // namespace ops

fc* fc::make(int in_dim, int out_dim, activation* activation)
{
  return new fc(in_dim, out_dim, activation);
}

void fc::ws(int size, const float* in, const float* params, float* out)
{
  /* transform raw pointers into thrust pointers */
  auto in_ptr = thrust::device_pointer_cast(in);
  auto params_ptr = thrust::device_pointer_cast(params);
  auto out_ptr = thrust::device_pointer_cast(out);

  /* base sequence to be transformed into indexes */
  auto seq = thrust::make_counting_iterator<int>(0);

  /* operators used in the next transfromation of the indexes */
  auto in_op = ops::fc::in_seq_op(get_in_dim(), get_out_dim());
  auto params_op = ops::fc::mod_op(weight_size());
  auto out_op = ops::fc::div_op(get_in_dim());

  /* transform sequences into indexes */
  auto in_idx = thrust::make_transform_iterator(seq, in_op);
  auto params_idx = thrust::make_transform_iterator(seq, params_op);
  auto out_idx = thrust::make_transform_iterator(seq, out_op);

  /* permutations of inputs and params for multiplication */
  auto in_perm = thrust::make_permutation_iterator(in_ptr, in_idx);
  auto params_perm = thrust::make_permutation_iterator(params_ptr, params_idx);

  /* zip inputs and params sequence for multiplication */
  auto tuple_in_params = thrust::make_tuple(in_perm, params_perm);
  auto ziped_in_params = thrust::make_zip_iterator(tuple_in_params);

  /* multiplication operation over ziped inputs and params */
  auto mult_op = ops::fc::mult();

  /* multiply inputs with params */
  auto mult_result = thrust::make_transform_iterator(ziped_in_params, mult_op);

  /* reduce multiplication iterators */
  auto begin_idx = out_idx;
  auto end_idx = out_idx + (size * get_in_dim() * get_out_dim());
  auto discard = thrust::make_discard_iterator();

  /* perform reduction */
  thrust::reduce_by_key(begin_idx, end_idx, mult_result, discard, out_ptr);

  /* add bias to the final result */
  auto bias_op = ops::fc::bias_op(weight_size(), bias_size());
  auto bias_idx = thrust::make_transform_iterator(seq, bias_op);
  auto bias_perm = thrust::make_permutation_iterator(params_ptr, bias_idx);

  auto begin_out = out_ptr;
  auto end_out = out_ptr + (get_out_dim() * size);
  auto sum_op = thrust::plus<float>();

  thrust::transform(begin_out, end_out, bias_perm, out_ptr, sum_op);
}

layer* fc::clone()
{
  return 0;
}

fc::fc(int in_dim, int out_dim, activation* activation)
: core::layers::fc(in_dim, out_dim, activation),
  core::layer(in_dim, out_dim, activation),
  layer(in_dim, out_dim, activation)
{

}

} // namespace layers
} // namespace cuda
} // namespace dnn_opt
