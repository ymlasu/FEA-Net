/// \file inner_product.cc
/// \author David Stutz
/// \brief Implementation of a inner product (i.e. fully connected layer)
/// operation in Tensorflow.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("MaskConv")
  .Input("input: float")
  .Input("weights: float")
  .Input("rho: float")
  .Output("mask_conv: float");

/// \brief Implementation of an inner product operation.
/// \param context
/// \author David Stutz
class MaskConvOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit MaskConvOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // some checks to be sure ...
    DCHECK_EQ(3, context->num_inputs());
    
    // get the input tensor
    const Tensor& input = context->input(0);
    
    // get the weight tensor
    const Tensor& weights = context->input(1);

    // get the heat conductivity
    const Tensor& rho = context->input(2);

    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weights.shape();
    
    // check input is a standing vector
    DCHECK_EQ(input_shape.dims(), 4);
    //DCHECK_EQ(input_shape.dim_size(1), input_shape.dim_size(2));
    
    // check weights is matrix of correct size
    DCHECK_EQ(weights_shape.dims(), 4);
    //DCHECK_EQ(input_shape.dim_size(2), weights_shape.dim_size(2));
    
    // create output shape
    TensorShape output_shape;
    output_shape.AddDim(input_shape.dim_size(0));
    output_shape.AddDim(input_shape.dim_size(1)-2);
    output_shape.AddDim(input_shape.dim_size(2)-2);
    output_shape.AddDim(input_shape.dim_size(3));            

    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    auto input_tensor = input.tensor<float, 4>();
    auto weights_tensor = weights.tensor<float, 4>();
    auto rho_tensor = rho.tensor<float, 1>();
    auto output_tensor = output->tensor<float, 4>();

    auto rho_1 = rho_tensor(0);
    auto rho_2 = rho_tensor(1);
    auto coef_1 = rho_1 / 3.;
    auto coef_2 = rho_2 / 3.;

    for (int i = 1; i < output->shape().dim_size(1)+1; i++) {
        for (int j = 1; j < output->shape().dim_size(2)+1; j++) {
            // from eq.11, mask_conv_expansion
            auto v_mat1 = input_tensor(0, i-1, j-1, 0) * weights_tensor(0, i-1, j-1, 0)
                                 + input_tensor(0, i-1, j+1, 0) * weights_tensor(0, i-1, j, 0)
                                 + input_tensor(0, i+1, j-1, 0) * weights_tensor(0, i, j-1, 0)
                                 + input_tensor(0, i+1, j+1, 0) * weights_tensor(0, i, j, 0)

                                 + input_tensor(0, i-1, j, 0) * (weights_tensor(0, i-1, j-1, 0) + weights_tensor(0, i-1, j, 0)) / 2.
                                 + input_tensor(0, i, j-1, 0) * (weights_tensor(0, i-1, j-1, 0) + weights_tensor(0, i, j-1, 0)) / 2.
                                 + input_tensor(0, i, j + 1, 0) * (weights_tensor(0, i-1, j, 0) + weights_tensor(0, i, j, 0)) / 2.
                                 + input_tensor(0, i+1, j, 0) * (weights_tensor(0, i, j-1, 0) + weights_tensor(0, i, j, 0)) / 2. ;

            auto v_mat2 = input_tensor(0, i-1, j-1, 0) * (1-weights_tensor(0, i-1, j-1, 0))
                                 + input_tensor(0, i-1, j+1, 0) * (1-weights_tensor(0, i-1, j, 0))
                                 + input_tensor(0, i+1, j-1, 0) * (1-weights_tensor(0, i, j-1, 0))
                                 + input_tensor(0, i+1, j+1, 0) * (1-weights_tensor(0, i, j, 0))

                                 + input_tensor(0, i-1, j, 0) * (2-weights_tensor(0, i-1, j-1, 0) - weights_tensor(0, i-1, j, 0)) / 2.
                                 + input_tensor(0, i, j-1, 0) * (2-weights_tensor(0, i-1, j-1, 0) - weights_tensor(0, i, j-1, 0)) / 2.
                                 + input_tensor(0, i, j + 1, 0) * (2-weights_tensor(0, i-1, j, 0) - weights_tensor(0, i, j, 0)) / 2.
                                 + input_tensor(0, i+1, j, 0) * (2-weights_tensor(0, i, j-1, 0) - weights_tensor(0, i, j, 0)) / 2. ;

            output_tensor(0, i-1, j-1, 0)  = v_mat1*coef_1 + v_mat2*coef_2;
             

        }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MaskConv").Device(DEVICE_CPU), MaskConvOp);
