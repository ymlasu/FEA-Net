/// \file inner_product_grad.cc
/// \author David Stutz
/// \brief Implementation of the gradient of a inner product operation, see
/// inner_product.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// the gradients are simply passed as additional arguments as
// they are available in the Python function for registering the gradient operation.
REGISTER_OP("MaskConvGrad")
  .Input("grad: float32")
  .Input("input: float32")
  .Input("weights: float32")
  .Input("rho: float32")
  .Output("grad_input: float32")
  .Output("grad_weights: float32")
  .Output("grad_rho: float32");

/// \brief Implementation of an inner product gradient operation.
/// Note that this operation is used in Python to register the gradient as
/// this is not possible in C*+ right now.
/// \param context
/// \author David Stutz
class MaskConvGradOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit MaskConvGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  /// \brief Compute the inner product gradients.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // output and grad is provided as input
    DCHECK_EQ(4, context->num_inputs());

    // get the gradient tensor
    const Tensor& grad = context->input(0);
    
    // get the original input tensor
    const Tensor& input = context->input(1);
    
    // get the weight tensor
    const Tensor& weights = context->input(2);

    // get the heat conductivity
    const Tensor& rho = context->input(3);

    // create input shape (inferred from the additional attribute `n`)
    TensorShape input_shape = input.shape();
    TensorShape weights_shape = weights.shape();
    TensorShape rho_shape = rho.shape();

    //DCHECK_EQ(input_shape.dim_size(0), weights_shape.dim_size(1));
    //DCHECK_EQ(weights_shape.dim_size(0), grad.shape().dim_size(0));
    
    // create output tensors
    Tensor* grad_input = NULL;
    Tensor* grad_weights = NULL;
    Tensor* grad_rho = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_weights));
    OP_REQUIRES_OK(context, context->allocate_output(2, rho_shape, &grad_rho));

    // get the Eigen tensors for data access
    auto grad_tensor = grad.tensor<float, 4>();
    auto weights_tensor = weights.tensor<float, 4>();
    auto input_tensor = input.tensor<float, 4>();
    auto rho_tensor = rho.tensor<float, 1>();

    auto grad_input_tensor = grad_input->tensor<float, 4>();   // response
    auto grad_weights_tensor = grad_weights->tensor<float, 4>(); // mask
    auto grad_rho_tensor = grad_rho->tensor<float, 1>(); // filter

    // zero initialization
    for (int i = 0; i < input_shape.dim_size(1); i++) {
      for (int j = 0; j < input_shape.dim_size(2); j++) {
          grad_input_tensor(0, i, j, 0) = 0.0;
      }
    }
    for (int i = 0; i < weights_shape.dim_size(1); i++) {
      for (int j = 0; j < weights_shape.dim_size(2); j++) {
          grad_weights_tensor(0, i, j, 0) = 0.0;
      }
    }
    grad_rho_tensor(0) = 0.0;
    grad_rho_tensor(1) = 0.0;

    auto rho_1 = rho_tensor(0);
    auto rho_2 = rho_tensor(1);
    auto diag_coef_1 = rho_1 / 3.;
    auto side_coef_1 = rho_1 / 3.;
    auto diag_coef_2 = rho_2 / 3.;
    auto side_coef_2 = rho_2 / 3.;
    auto diag_coef_diff = diag_coef_1 - diag_coef_2;
    auto side_coef_diff = side_coef_1 - side_coef_2;

    auto dbg = 0;
    if (dbg){
        std::cout<<"rho1:  "<<rho_1<<"  rho2:  "<<rho_2<<std::endl;
        std::cout<<weights_shape.dim_size(0)<<" "<<weights_shape.dim_size(1)<<" "<<weights_shape.dim_size(2)<<" "<<weights_shape.dim_size(3)<<std::endl;
        std::cout<<"grad_tensor"<<std::endl;
        std::cout<<grad.dim_size(0)<<" "<<grad.dim_size(1)<<" "<<grad.dim_size(2)<<" "<<grad.dim_size(3)<<std::endl;
    }

    for (int i = 1; i < weights_shape.dim_size(1); i++) {
      for (int j = 1; j < weights_shape.dim_size(2); j++) {
            grad_weights_tensor(0,i-1,j-1,0) +=  grad_tensor(0,i-1,j-1,0) * (input_tensor(0,i-1,j-1,0) * diag_coef_diff + (input_tensor(0,i-1,j,0) + input_tensor(0,i,j-1,0))/2. * side_coef_diff);
            grad_weights_tensor(0,i-1,j,0) += grad_tensor(0,i-1,j-1,0)* (input_tensor(0,i-1,j+1,0) * diag_coef_diff + (input_tensor(0,i,j+1,0) + input_tensor(0,i-1,j,0))/ 2. * side_coef_diff);
            grad_weights_tensor(0,i,j-1,0) += grad_tensor(0,i-1,j-1,0) *  (input_tensor(0,i+1,j-1,0) * diag_coef_diff + (input_tensor(0,i+1,j,0) + input_tensor(0,i,j-1,0))/ 2. * side_coef_diff);
            grad_weights_tensor(0,i,j,0) += grad_tensor(0,i-1,j-1,0) *  (input_tensor(0,i+1,j+1,0) * diag_coef_diff + (input_tensor(0,i+1,j,0) + input_tensor(0,i,j+1,0))/ 2. * side_coef_diff) ;
      }
    }

    for (int i = 1; i < input_shape.dim_size(1)-1; i++) {
      for (int j = 1; j < input_shape.dim_size(2)-1; j++) {
            grad_input_tensor(0,i-1,j-1,0) +=  grad_tensor(0,i-1,j-1,0) * (weights_tensor(0,i-1,j-1,0) * diag_coef_diff + diag_coef_2);
            grad_input_tensor(0,i-1,j,0) += grad_tensor(0,i-1,j-1,0) * ((weights_tensor(0, i-1, j-1, 0)+weights_tensor(0, i-1, j, 0))/2 * side_coef_diff + side_coef_2);
            grad_input_tensor(0,i-1,j+1,0) += grad_tensor(0,i-1,j-1,0) * (weights_tensor(0, i-1, j, 0) * diag_coef_diff + diag_coef_2);

            grad_input_tensor(0,i,j-1,0) += grad_tensor(0,i-1,j-1,0) * ((weights_tensor(0, i-1, j-1, 0)+weights_tensor(0, i, j-1, 0))/2 * side_coef_diff + side_coef_2);
            grad_input_tensor(0,i,j+1,0) += grad_tensor(0,i-1,j-1,0) * ((weights_tensor(0, i-1, j, 0)+weights_tensor(0, i, j, 0))/2 * side_coef_diff + side_coef_2);

            grad_input_tensor(0,i+1,j-1,0) += grad_tensor(0,i-1,j-1,0) * (weights_tensor(0, i, j-1, 0) * diag_coef_diff + diag_coef_2);
            grad_input_tensor(0,i+1,j,0) +=  grad_tensor(0,i-1,j-1,0) * ((weights_tensor(0, i, j-1, 0)+weights_tensor(0, i, j, 0))/2 * side_coef_diff + side_coef_2);
            grad_input_tensor(0,i+1,j+1,0) += grad_tensor(0,i-1,j-1,0) *  (weights_tensor(0, i, j, 0) * diag_coef_diff + diag_coef_2 );
      }
    }

    for (int i = 1; i < input_shape.dim_size(1)-1; i++) {
      for (int j = 1; j < input_shape.dim_size(2)-1; j++) {
            grad_rho_tensor(0) +=  grad_tensor(0,i-1,j-1,0) * (input_tensor(0,i-1,j-1,0)*weights_tensor(0,i-1,j-1,0)
                                                              +input_tensor(0,i-1,j+1,0)*weights_tensor(0,i-1,j,0)
                                                              +input_tensor(0,i+1,j-1,0)*weights_tensor(0,i,j-1,0)
                                                              +input_tensor(0,i+1,j+1,0)*weights_tensor(0,i,j,0)

                                                              +input_tensor(0,i-1,j,0)*(weights_tensor(0,i-1,j-1,0)+weights_tensor(0,i-1,j,0))/2.
                                                              +input_tensor(0,i,j-1,0)*(weights_tensor(0,i-1,j-1,0)+weights_tensor(0,i,j-1,0))/2.
                                                              +input_tensor(0,i,j+1,0)*(weights_tensor(0,i-1,j,0)+weights_tensor(0,i,j,0))/2.
                                                              +input_tensor(0,i+1,j,0)*(weights_tensor(0,i,j-1,0)+weights_tensor(0,i,j,0))/2.
                                                              )/3.;

            grad_rho_tensor(1) +=  grad_tensor(0,i-1,j-1,0) * (input_tensor(0,i-1,j-1,0)*(1-weights_tensor(0,i-1,j-1,0))
                                                              +input_tensor(0,i-1,j+1,0)*(1-weights_tensor(0,i-1,j,0))
                                                              +input_tensor(0,i+1,j-1,0)*(1-weights_tensor(0,i,j-1,0))
                                                              +input_tensor(0,i+1,j+1,0)*(1-weights_tensor(0,i,j,0))

                                                              +input_tensor(0,i-1,j,0)*(2-weights_tensor(0,i-1,j-1,0)-weights_tensor(0,i-1,j,0))/2.
                                                              +input_tensor(0,i,j-1,0)*(2-weights_tensor(0,i-1,j-1,0)-weights_tensor(0,i,j-1,0))/2.
                                                              +input_tensor(0,i,j+1,0)*(2-weights_tensor(0,i-1,j,0)-weights_tensor(0,i,j,0))/2.
                                                              +input_tensor(0,i+1,j,0)*(2-weights_tensor(0,i,j-1,0)-weights_tensor(0,i,j,0))/2.
                                                              )/3.;
      }
    }

  }
};

REGISTER_KERNEL_BUILDER(Name("MaskConvGrad").Device(DEVICE_CPU), MaskConvGradOp);
