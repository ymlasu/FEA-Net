/// \file inner_product_grad.cc
/// \author David Stutz
/// \brief Implementation of the gradient of a inner product operation, see
/// inner_product.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// the gradients are simply passed as additional arguments as
// they are available in the Python function for registering the gradient operation.
REGISTER_OP("MaskconvElastGrad")
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
class MaskconvElastGradOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit MaskconvElastGradOp(OpKernelConstruction* context) : OpKernel(context) {
    
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
    grad_rho_tensor(2) = 0.0;
    grad_rho_tensor(3) = 0.0;

    auto E_1 = rho_tensor(0);
    auto mu_1 = rho_tensor(1);
    auto E_2 = rho_tensor(2);
    auto mu_2 = rho_tensor(3);
    auto coef_1 = E_1 / (16. * (1-mu_1*mu_1));
    auto coef_2 = E_2 / (16. * (1-mu_2*mu_2));

    // k1_xx, diagonal
    auto k1_xx_00 = 8-8/3.*mu_1;
    auto k1_xx_11 = k1_xx_00;
    auto k1_xx_22 = k1_xx_00;
    auto k1_xx_33 = k1_xx_00;
    // k1_xx, off-diagonal
    auto k1_xx_01 = -4/3.*mu_1 - 4.;
    auto k1_xx_10 = k1_xx_01;
    auto k1_xx_02 = 4/3.*mu_1 - 4.;
    auto k1_xx_20 = k1_xx_02;
    auto k1_xx_03 = 8/3. * mu_1;
    auto k1_xx_30 = k1_xx_03;
    auto k1_xx_12 = 8/3. * mu_1;
    auto k1_xx_21 = k1_xx_12;
    auto k1_xx_13 = 4/3.*mu_1 - 4;
    auto k1_xx_31 = k1_xx_13;
    auto k1_xx_23 = -4/3.*mu_1 - 4;
    auto k1_xx_32 = k1_xx_23;

    // k1_yy, diagonal
    auto k1_yy_00 = 8-8/3.*mu_1;
    auto k1_yy_11 = k1_yy_00;
    auto k1_yy_22 = k1_yy_00;
    auto k1_yy_33 = k1_yy_00;
    // k1_yy, off-diagonal
    auto k1_yy_01 = 8/3. * mu_1;
    auto k1_yy_10 = k1_yy_01;
    auto k1_yy_02 = 4/3.*mu_1 - 4.;
    auto k1_yy_20 = k1_yy_02;
    auto k1_yy_03 = -4/3.*mu_1 - 4.;
    auto k1_yy_30 = k1_yy_03;
    auto k1_yy_12 = -4/3.*mu_1 - 4;
    auto k1_yy_21 = k1_yy_12;
    auto k1_yy_13 = 4/3.*mu_1 - 4;
    auto k1_yy_31 = k1_yy_13;
    auto k1_yy_23 = 8/3. * mu_1;
    auto k1_yy_32 = k1_yy_23;

    // k1_yx, diagonal
    auto k1_yx_00 = 2*mu_1+2;
    auto k1_yx_11 = -k1_yx_00;
    auto k1_yx_22 = k1_yx_00;
    auto k1_yx_33 = -k1_yx_00;
    // k1_yx, off-diagonal
    auto k1_yx_01 = 2-6*mu_1;
    auto k1_yx_10 = -k1_yx_01;
    auto k1_yx_02 = -2*mu_1-2;
    auto k1_yx_20 = k1_yx_02;
    auto k1_yx_03 = 6*mu_1-2;
    auto k1_yx_30 = -k1_yx_03;
    auto k1_yx_12 = 2-6*mu_1;
    auto k1_yx_21 = -k1_yx_12;
    auto k1_yx_13 = 2*mu_1+2;
    auto k1_yx_31 = k1_yx_13;
    auto k1_yx_23 = 2-6*mu_1;
    auto k1_yx_32 = -k1_yx_23;

    // k1_xy, diagonal
    auto k1_xy_00 = 2*mu_1+2;
    auto k1_xy_11 = -k1_xy_00;
    auto k1_xy_22 = k1_xy_00;
    auto k1_xy_33 = -k1_xy_00;
    // k1_xy, off-diagonal
    auto k1_xy_01 = 6*mu_1-2;
    auto k1_xy_10 = -k1_xy_01;
    auto k1_xy_02 = -2*mu_1-2;
    auto k1_xy_20 = k1_xy_02;
    auto k1_xy_03 = 2-6*mu_1;
    auto k1_xy_30 = -k1_xy_03;
    auto k1_xy_12 = 6*mu_1-2;
    auto k1_xy_21 = -k1_xy_12;
    auto k1_xy_13 = 2*mu_1+2;
    auto k1_xy_31 = k1_xy_13;
    auto k1_xy_23 = 6*mu_1-2;
    auto k1_xy_32 = -k1_xy_23;

    // k2_xx, diagonal
    auto k2_xx_00 = 8-8/3.*mu_2;
    auto k2_xx_11 = k2_xx_00;
    auto k2_xx_22 = k2_xx_00;
    auto k2_xx_33 = k2_xx_00;
    // k2_xx, off-diagonal
    auto k2_xx_01 = -4/3.*mu_2 - 4.;
    auto k2_xx_10 = k2_xx_01;
    auto k2_xx_02 = 4/3.*mu_2 - 4.;
    auto k2_xx_20 = k2_xx_02;
    auto k2_xx_03 = 8/3. * mu_2;
    auto k2_xx_30 = k2_xx_03;
    auto k2_xx_12 = 8/3. * mu_2;
    auto k2_xx_21 = k2_xx_12;
    auto k2_xx_13 = 4/3.*mu_2 - 4.;
    auto k2_xx_31 = k2_xx_13;
    auto k2_xx_23 = -4/3.*mu_2 - 4;
    auto k2_xx_32 = k2_xx_23;

    // k2_yy, diagonal
    auto k2_yy_00 = 8-8/3.*mu_2;
    auto k2_yy_11 = k2_yy_00;
    auto k2_yy_22 = k2_yy_00;
    auto k2_yy_33 = k2_yy_00;
    // k2_yy, off-diagonal, PROBLEMATIC!!
    auto k2_yy_01 = 8/3. * mu_2;
    auto k2_yy_10 = k2_yy_01;
    auto k2_yy_02 = 4/3.*mu_2 - 4.;
    auto k2_yy_20 = k2_yy_02;
    auto k2_yy_03 = -4/3.*mu_2 - 4.;
    auto k2_yy_30 = k2_yy_03;
    auto k2_yy_12 = -4/3.*mu_2 - 4;
    auto k2_yy_21 = k2_yy_12;
    auto k2_yy_13 = 4/3.*mu_2 - 4;
    auto k2_yy_31 = k2_yy_13;
    auto k2_yy_23 = 8/3. * mu_2;
    auto k2_yy_32 = k2_yy_23;

    // k2_yx, diagonal
    auto k2_yx_00 = 2*mu_2+2;
    auto k2_yx_11 = -k2_yx_00;
    auto k2_yx_22 = k2_yx_00;
    auto k2_yx_33 = -k2_yx_00;
    // k2_yx, off-diagonal
    auto k2_yx_01 = 2-6*mu_2;
    auto k2_yx_10 = -k2_yx_01;
    auto k2_yx_02 = -2*mu_2-2;
    auto k2_yx_20 = k2_yx_02;
    auto k2_yx_03 = 6*mu_2-2;
    auto k2_yx_30 = -k2_yx_03;
    auto k2_yx_12 = 2-6*mu_2;
    auto k2_yx_21 = -k2_yx_12;
    auto k2_yx_13 = 2*mu_2+2;
    auto k2_yx_31 = k2_yx_13;
    auto k2_yx_23 = 2-6*mu_2;
    auto k2_yx_32 = -k2_yx_23;

    // k2_xy, diagonal
    auto k2_xy_00 = 2*mu_2+2;
    auto k2_xy_11 = -k2_xy_00;
    auto k2_xy_22 = k2_xy_00;
    auto k2_xy_33 = -k2_xy_00;
    // k2_xy, off-diagonal
    auto k2_xy_01 = 6*mu_2-2;
    auto k2_xy_10 = -k2_xy_01;
    auto k2_xy_02 = -2*mu_2-2;
    auto k2_xy_20 = k2_xy_02;
    auto k2_xy_03 = 2-6*mu_2;
    auto k2_xy_30 = -k2_xy_03;
    auto k2_xy_12 = 6*mu_2-2;
    auto k2_xy_21 = -k2_xy_12;
    auto k2_xy_13 = 2*mu_2+2;
    auto k2_xy_31 = k2_xy_13;
    auto k2_xy_23 = 6*mu_2-2;
    auto k2_xy_32 = -k2_xy_23;

    //
    //_coef_grad_mu
    //
        // k1_xx, diagonal
    auto coef_1_grad_mu = E_1 *mu_1 / (8. * (1-mu_1*mu_1)*(1-mu_1*mu_1));
    auto coef_2_grad_mu = E_2 *mu_2 / (8. * (1-mu_2*mu_2)*(1-mu_2*mu_2));

    auto k1_xx_00_coef_grad_mu = -8/3.*coef_1 + k1_xx_00*coef_1_grad_mu;
    auto k1_xx_11_coef_grad_mu = k1_xx_00_coef_grad_mu;
    auto k1_xx_22_coef_grad_mu = k1_xx_00_coef_grad_mu;
    auto k1_xx_33_coef_grad_mu = k1_xx_00_coef_grad_mu;
    // k1_xx, off-diagonal
    auto k1_xx_01_coef_grad_mu = -4/3.*coef_1 + k1_xx_01*coef_1_grad_mu;
    auto k1_xx_10_coef_grad_mu = k1_xx_01_coef_grad_mu;
    auto k1_xx_02_coef_grad_mu = 4/3.*coef_1 + k1_xx_02*coef_1_grad_mu;
    auto k1_xx_20_coef_grad_mu = k1_xx_02_coef_grad_mu;
    auto k1_xx_03_coef_grad_mu = 8/3.*coef_1 + k1_xx_03*coef_1_grad_mu;
    auto k1_xx_30_coef_grad_mu = k1_xx_03_coef_grad_mu;
    auto k1_xx_12_coef_grad_mu = 8/3.*coef_1 + k1_xx_12*coef_1_grad_mu;
    auto k1_xx_21_coef_grad_mu = k1_xx_12_coef_grad_mu;
    auto k1_xx_13_coef_grad_mu = 4/3.*coef_1 + k1_xx_13*coef_1_grad_mu;
    auto k1_xx_31_coef_grad_mu = k1_xx_13_coef_grad_mu;
    auto k1_xx_23_coef_grad_mu = -4/3.*coef_1 + k1_xx_23*coef_1_grad_mu;
    auto k1_xx_32_coef_grad_mu = k1_xx_23_coef_grad_mu;

    // k1_yy, diagonal
    auto k1_yy_00_coef_grad_mu = -8/3.*coef_1 + k1_yy_00*coef_1_grad_mu;
    auto k1_yy_11_coef_grad_mu = k1_yy_00_coef_grad_mu;
    auto k1_yy_22_coef_grad_mu = k1_yy_00_coef_grad_mu;
    auto k1_yy_33_coef_grad_mu = k1_yy_00_coef_grad_mu;
    // k1_yy, off-diagonal
    auto k1_yy_01_coef_grad_mu = 8/3.*coef_1 + k1_yy_01*coef_1_grad_mu;
    auto k1_yy_10_coef_grad_mu = k1_yy_01_coef_grad_mu;
    auto k1_yy_02_coef_grad_mu = 4/3.*coef_1 + k1_yy_02*coef_1_grad_mu;
    auto k1_yy_20_coef_grad_mu = k1_yy_02_coef_grad_mu;
    auto k1_yy_03_coef_grad_mu = -4/3.*coef_1 + k1_yy_03*coef_1_grad_mu;
    auto k1_yy_30_coef_grad_mu = k1_yy_03_coef_grad_mu;
    auto k1_yy_12_coef_grad_mu = -4/3.*coef_1 + k1_yy_12*coef_1_grad_mu;
    auto k1_yy_21_coef_grad_mu = k1_yy_12_coef_grad_mu;
    auto k1_yy_13_coef_grad_mu = 4/3.*coef_1 + k1_yy_13*coef_1_grad_mu;
    auto k1_yy_31_coef_grad_mu = k1_yy_13_coef_grad_mu;
    auto k1_yy_23_coef_grad_mu = 8/3.*coef_1 + k1_yy_23*coef_1_grad_mu;
    auto k1_yy_32_coef_grad_mu = k1_yy_23_coef_grad_mu;

    // k1_yx, diagonal
    auto k1_yx_00_coef_grad_mu = 2*coef_1 + k1_yx_00*coef_1_grad_mu;
    auto k1_yx_11_coef_grad_mu = -k1_yx_00_coef_grad_mu;
    auto k1_yx_22_coef_grad_mu = k1_yx_00_coef_grad_mu;
    auto k1_yx_33_coef_grad_mu = -k1_yx_00_coef_grad_mu;
    // k1_yx, off-diagonal
    auto k1_yx_01_coef_grad_mu = -6*coef_1 + k1_yx_01*coef_1_grad_mu;
    auto k1_yx_10_coef_grad_mu = -k1_yx_01_coef_grad_mu;
    auto k1_yx_02_coef_grad_mu = -2*coef_1 + k1_yx_02*coef_1_grad_mu;
    auto k1_yx_20_coef_grad_mu = k1_yx_02_coef_grad_mu;
    auto k1_yx_03_coef_grad_mu = 6*coef_1 + k1_yx_03*coef_1_grad_mu;
    auto k1_yx_30_coef_grad_mu = -k1_yx_03_coef_grad_mu;
    auto k1_yx_12_coef_grad_mu = -6*coef_1 + k1_yx_12*coef_1_grad_mu;
    auto k1_yx_21_coef_grad_mu = -k1_yx_12_coef_grad_mu;
    auto k1_yx_13_coef_grad_mu = 2*coef_1 + k1_yx_13*coef_1_grad_mu;
    auto k1_yx_31_coef_grad_mu = k1_yx_13_coef_grad_mu;
    auto k1_yx_23_coef_grad_mu = -6*coef_1 + k1_yx_23*coef_1_grad_mu;
    auto k1_yx_32_coef_grad_mu = -k1_yx_23_coef_grad_mu;

    // k1_xy, diagonal
    auto k1_xy_00_coef_grad_mu = 2*coef_1 + k1_xy_00*coef_1_grad_mu;
    auto k1_xy_11_coef_grad_mu = -k1_xy_00_coef_grad_mu;
    auto k1_xy_22_coef_grad_mu = k1_xy_00_coef_grad_mu;
    auto k1_xy_33_coef_grad_mu = -k1_xy_00_coef_grad_mu;
    // k1_xy, off-diagonal
    auto k1_xy_01_coef_grad_mu = 6*coef_1 + k1_xy_01*coef_1_grad_mu;
    auto k1_xy_10_coef_grad_mu = -k1_xy_01_coef_grad_mu;
    auto k1_xy_02_coef_grad_mu = -2*coef_1 + k1_xy_02*coef_1_grad_mu;
    auto k1_xy_20_coef_grad_mu = k1_xy_02_coef_grad_mu;
    auto k1_xy_03_coef_grad_mu = -6*coef_1 + k1_xy_03*coef_1_grad_mu;
    auto k1_xy_30_coef_grad_mu = -k1_xy_03_coef_grad_mu;
    auto k1_xy_12_coef_grad_mu = 6*coef_1 + k1_xy_12*coef_1_grad_mu;
    auto k1_xy_21_coef_grad_mu = -k1_xy_12_coef_grad_mu;
    auto k1_xy_13_coef_grad_mu = 2*coef_1 + k1_xy_13*coef_1_grad_mu;
    auto k1_xy_31_coef_grad_mu = k1_xy_13_coef_grad_mu;
    auto k1_xy_23_coef_grad_mu = 6*coef_1 + k1_xy_23*coef_1_grad_mu;
    auto k1_xy_32_coef_grad_mu = -k1_xy_23_coef_grad_mu;

    // k2_xx, diagonal
    auto k2_xx_00_coef_grad_mu = -8/3.*coef_2 + k2_xx_00*coef_2_grad_mu;
    auto k2_xx_11_coef_grad_mu = k2_xx_00;
    auto k2_xx_22_coef_grad_mu = k2_xx_00;
    auto k2_xx_33_coef_grad_mu = k2_xx_00;
    // k2_xx, off-diagonal
    auto k2_xx_01_coef_grad_mu = -4/3.*coef_2 + k2_xx_01*coef_2_grad_mu;
    auto k2_xx_10_coef_grad_mu = k2_xx_01_coef_grad_mu;
    auto k2_xx_02_coef_grad_mu = 4/3.*coef_2 + k2_xx_02*coef_2_grad_mu;
    auto k2_xx_20_coef_grad_mu = k2_xx_02_coef_grad_mu;
    auto k2_xx_03_coef_grad_mu = 8/3.*coef_2 + k2_xx_03*coef_2_grad_mu;
    auto k2_xx_30_coef_grad_mu = k2_xx_03_coef_grad_mu;
    auto k2_xx_12_coef_grad_mu = 8/3.*coef_2 + k2_xx_12*coef_2_grad_mu;
    auto k2_xx_21_coef_grad_mu = k2_xx_12_coef_grad_mu;
    auto k2_xx_13_coef_grad_mu = 4/3.*coef_2 + k2_xx_13*coef_2_grad_mu;
    auto k2_xx_31_coef_grad_mu = k2_xx_13_coef_grad_mu;
    auto k2_xx_23_coef_grad_mu = -4/3.*coef_2 + k2_xx_23*coef_2_grad_mu;
    auto k2_xx_32_coef_grad_mu = k2_xx_23_coef_grad_mu;

    // k2_yy, diagonal
    auto k2_yy_00_coef_grad_mu = -8/3.*coef_2 + k2_yy_00*coef_2_grad_mu;
    auto k2_yy_11_coef_grad_mu = k2_yy_00_coef_grad_mu;
    auto k2_yy_22_coef_grad_mu = k2_yy_00_coef_grad_mu;
    auto k2_yy_33_coef_grad_mu = k2_yy_00_coef_grad_mu;
    // k2_yy, off-diagonal, PROBLEMATIC!!
    auto k2_yy_01_coef_grad_mu = 8/3.*coef_2 + k2_yy_01*coef_2_grad_mu;
    auto k2_yy_10_coef_grad_mu = k2_yy_01_coef_grad_mu;
    auto k2_yy_02_coef_grad_mu = 4/3.*coef_2 + k2_yy_02*coef_2_grad_mu;
    auto k2_yy_20_coef_grad_mu = k2_yy_02_coef_grad_mu;
    auto k2_yy_03_coef_grad_mu = -4/3.*coef_2 + k2_yy_03*coef_2_grad_mu;
    auto k2_yy_30_coef_grad_mu = k2_yy_03_coef_grad_mu;
    auto k2_yy_12_coef_grad_mu = -4/3.*coef_2 + k2_yy_12*coef_2_grad_mu;
    auto k2_yy_21_coef_grad_mu = k2_yy_12_coef_grad_mu;
    auto k2_yy_13_coef_grad_mu = 4/3.*coef_2 + k2_yy_13*coef_2_grad_mu;
    auto k2_yy_31_coef_grad_mu = k2_yy_13_coef_grad_mu;
    auto k2_yy_23_coef_grad_mu = 8/3. *coef_2 + k2_yy_23*coef_2_grad_mu;
    auto k2_yy_32_coef_grad_mu = k2_yy_23_coef_grad_mu;

    // k2_yx, diagonal
    auto k2_yx_00_coef_grad_mu = 2*coef_2 + k2_yx_00*coef_2_grad_mu;
    auto k2_yx_11_coef_grad_mu = -k2_yx_00_coef_grad_mu;
    auto k2_yx_22_coef_grad_mu = k2_yx_00_coef_grad_mu;
    auto k2_yx_33_coef_grad_mu = -k2_yx_00_coef_grad_mu;
    // k2_yx, off-diagonal
    auto k2_yx_01_coef_grad_mu = -6*coef_2 + k2_yx_01*coef_2_grad_mu;
    auto k2_yx_10_coef_grad_mu = -k2_yx_01_coef_grad_mu;
    auto k2_yx_02_coef_grad_mu = -2*coef_2 + k2_yx_02*coef_2_grad_mu;
    auto k2_yx_20_coef_grad_mu = k2_yx_02_coef_grad_mu;
    auto k2_yx_03_coef_grad_mu = 6*coef_2 + k2_yx_03*coef_2_grad_mu;
    auto k2_yx_30_coef_grad_mu = -k2_yx_03_coef_grad_mu;
    auto k2_yx_12_coef_grad_mu = -6*coef_2 + k2_yx_12*coef_2_grad_mu;
    auto k2_yx_21_coef_grad_mu = -k2_yx_12_coef_grad_mu;
    auto k2_yx_13_coef_grad_mu = 2*coef_2 + k2_yx_13*coef_2_grad_mu;
    auto k2_yx_31_coef_grad_mu = k2_yx_13_coef_grad_mu;
    auto k2_yx_23_coef_grad_mu = -6*coef_2 + k2_yx_23*coef_2_grad_mu;
    auto k2_yx_32_coef_grad_mu = -k2_yx_23_coef_grad_mu;

    // k2_xy, diagonal
    auto k2_xy_00_coef_grad_mu = 2*coef_2 + k2_xy_00*coef_2_grad_mu;
    auto k2_xy_11_coef_grad_mu = -k2_xy_00_coef_grad_mu;
    auto k2_xy_22_coef_grad_mu = k2_xy_00_coef_grad_mu;
    auto k2_xy_33_coef_grad_mu = -k2_xy_00_coef_grad_mu;
    // k2_xy, off-diagonal
    auto k2_xy_01_coef_grad_mu = 6*coef_2 + k2_xy_01*coef_2_grad_mu;
    auto k2_xy_10_coef_grad_mu = -k2_xy_01_coef_grad_mu;
    auto k2_xy_02_coef_grad_mu = -2*coef_2 + k2_xy_02*coef_2_grad_mu;
    auto k2_xy_20_coef_grad_mu = k2_xy_02_coef_grad_mu;
    auto k2_xy_03_coef_grad_mu = -6*coef_2 + k2_xy_03*coef_2_grad_mu;
    auto k2_xy_30_coef_grad_mu = -k2_xy_03_coef_grad_mu;
    auto k2_xy_12_coef_grad_mu = 6*coef_2 + k2_xy_12*coef_2_grad_mu;
    auto k2_xy_21_coef_grad_mu = -k2_xy_12_coef_grad_mu;
    auto k2_xy_13_coef_grad_mu = 2*coef_2 + k2_xy_13*coef_2_grad_mu;
    auto k2_xy_31_coef_grad_mu = k2_xy_13_coef_grad_mu;
    auto k2_xy_23_coef_grad_mu = 6*coef_2 + k2_xy_23*coef_2_grad_mu;
    auto k2_xy_32_coef_grad_mu = -k2_xy_23_coef_grad_mu;

//    auto dbg = 0;
//    if (dbg){
//        std::cout<<"rho1:  "<<rho_1<<"  rho2:  "<<rho_2<<std::endl;
//        std::cout<<weights_shape.dim_size(0)<<" "<<weights_shape.dim_size(1)<<" "<<weights_shape.dim_size(2)<<" "<<weights_shape.dim_size(3)<<std::endl;
//        std::cout<<"grad_tensor"<<std::endl;
//        std::cout<<grad.dim_size(0)<<" "<<grad.dim_size(1)<<" "<<grad.dim_size(2)<<" "<<grad.dim_size(3)<<std::endl;
//    }

    for (int i = 1; i < weights_shape.dim_size(1); i++) {
      for (int j = 1; j < weights_shape.dim_size(2); j++) {
//      grad_tensor(0,i-1,j-1,0) *
            grad_weights_tensor(0,i-1,j-1,0) += grad_tensor(0,i-1,j-1,0) *
                                            ((
                                            k1_xx_10*input_tensor(0, i, j-1, 0)
                                //        +k1_xx_11*input_tensor(0, i, j, 0)
                                            +k1_xx_12*input_tensor(0, i-1, j, 0)
                                            +k1_xx_13*input_tensor(0, i-1, j-1, 0)
                                            +k1_xy_10*input_tensor(0, i, j-1, 1)
                                            +k1_xy_11*input_tensor(0, i, j, 1)
                                            +k1_xy_12*input_tensor(0, i-1, j, 1)
                                            +k1_xy_13*input_tensor(0, i-1, j-1, 1)
                                            )*coef_1

                                            +(
                                            k1_yx_10*input_tensor(0, i, j-1, 0)
                                            +k1_yx_11*input_tensor(0, i, j, 0)
                                            +k1_yx_12*input_tensor(0, i-1, j, 0)
                                            +k1_yx_13*input_tensor(0, i-1, j-1, 0)
                                            +k1_yy_10*input_tensor(0, i, j-1, 1)
                                //        +k1_yy_11*input_tensor(0, i, j, 1)
                                            +k1_yy_12*input_tensor(0, i-1, j, 1)
                                            +k1_yy_13*input_tensor(0, i-1, j-1, 1)
                                             ) *coef_1

                                            +(
                                            k2_xx_10*input_tensor(0, i, j-1, 0)
                                //        +k2_xx_11*input_tensor(0, i, j, 0)
                                            +k2_xx_12*input_tensor(0, i-1, j, 0)
                                            +k2_xx_13*input_tensor(0, i-1, j-1, 0)
                                            +k2_xy_10*input_tensor(0, i, j-1, 1)
                                            +k2_xy_11*input_tensor(0, i, j, 1)
                                            +k2_xy_12*input_tensor(0, i-1, j, 1)
                                            +k2_xy_13*input_tensor(0, i-1, j-1, 1)
                                            ) *coef_2

                                            -(
                                            k2_yx_10*input_tensor(0, i, j-1, 0)
                                            +k2_yx_11*input_tensor(0, i, j, 0)
                                            +k2_yx_12*input_tensor(0, i-1, j, 0)
                                            +k2_yx_13*input_tensor(0, i-1, j-1, 0)
                                            +k2_yy_10*input_tensor(0, i, j-1, 1)
                                //        +k2_yy_11*input_tensor(0, i, j, 1)
                                            +k2_yy_12*input_tensor(0, i-1, j, 1)
                                            +k2_yy_13*input_tensor(0, i-1, j-1, 1)
                                            ) *coef_2);

            grad_weights_tensor(0,i-1,j,0) += grad_tensor(0,i-1,j,0) *
                                            ((
                                //        k1_xx_00*input_tensor(0, i, j, 0)
                                            +k1_xx_01*input_tensor(0, i, j+1, 0)
                                            +k1_xx_02*input_tensor(0, i-1, j+1, 0)
                                            +k1_xx_03*input_tensor(0, i-1, j, 0)
                                            +k1_xy_00*input_tensor(0, i, j, 1)
                                            +k1_xy_01*input_tensor(0, i, j+1, 1)
                                            +k1_xy_02*input_tensor(0, i-1, j+1, 1)
                                            +k1_xy_03*input_tensor(0, i-1, j, 1)
                                            ) *coef_1

                                            +(
                                            k1_yx_00*input_tensor(0, i, j, 0)
                                            +k1_yx_01*input_tensor(0, i, j+1, 0)
                                            +k1_yx_02*input_tensor(0, i-1, j+1, 0)
                                            +k1_yx_03*input_tensor(0, i-1, j, 0)
                                //        +k1_yy_00*input_tensor(0, i, j, 1)
                                            +k1_yy_01*input_tensor(0, i, j+1, 1)
                                            +k1_yy_02*input_tensor(0, i-1, j+1, 1)
                                            +k1_yy_03*input_tensor(0, i-1, j, 1)
                                            ) *coef_1

                                            -(
                                //        k2_xx_00*input_tensor(0, i, j, 0)
                                            +k2_xx_01*input_tensor(0, i, j+1, 0)
                                            +k2_xx_02*input_tensor(0, i-1, j+1, 0)
                                            +k2_xx_03*input_tensor(0, i-1, j, 0)
                                            +k2_xy_00*input_tensor(0, i, j, 1)
                                            +k2_xy_01*input_tensor(0, i, j+1, 1)
                                            +k2_xy_02*input_tensor(0, i-1, j+1, 1)
                                            +k2_xy_03*input_tensor(0, i-1, j, 1)
                                            ) *coef_2

                                            -(
                                            k2_yx_00*input_tensor(0, i, j, 0)
                                            +k2_yx_01*input_tensor(0, i, j+1, 0)
                                            +k2_yx_02*input_tensor(0, i-1, j+1, 0)
                                            +k2_yx_03*input_tensor(0, i-1, j, 0)
                                //        +k2_yy_00*input_tensor(0, i, j, 1)
                                            +k2_yy_01*input_tensor(0, i, j+1, 1)
                                            +k2_yy_02*input_tensor(0, i-1, j+1, 1)
                                            +k2_yy_03*input_tensor(0, i-1, j, 1)
                                            ) *coef_2);

            grad_weights_tensor(0,i,j-1,0) += grad_tensor(0,i,j-1,0) *
                                            ((
                                            k1_xx_20*input_tensor(0, i+1, j-1, 0)
                                            +k1_xx_21*input_tensor(0, i+1, j, 0)
                                //        +k1_xx_22*input_tensor(0, i, j, 0)
                                            +k1_xx_23*input_tensor(0, i, j-1, 0)
                                            +k1_xy_20*input_tensor(0, i+1, j-1, 1)
                                            +k1_xy_21*input_tensor(0, i+1, j, 1)
                                            +k1_xy_22*input_tensor(0, i, j, 1)
                                            +k1_xy_23*input_tensor(0, i, j-1, 1)
                                            ) *coef_1

                                            +(
                                            k1_yx_20*input_tensor(0, i+1, j-1, 0)
                                            +k1_yx_21*input_tensor(0, i+1, j, 0)
                                            +k1_yx_22*input_tensor(0, i, j, 0)
                                            +k1_yx_23*input_tensor(0, i, j-1, 0)
                                            +k1_yy_20*input_tensor(0, i+1, j-1, 1)
                                            +k1_yy_21*input_tensor(0, i+1, j, 1)
                                //        +k1_yy_22*input_tensor(0, i, j, 1)
                                            +k1_yy_23*input_tensor(0, i, j-1, 1)
                                            ) *coef_1

                                            -(
                                            k2_xx_20*input_tensor(0, i+1, j-1, 0)
                                            +k2_xx_21*input_tensor(0, i+1, j, 0)
                                //        +k2_xx_22*input_tensor(0, i, j, 0)
                                            +k2_xx_23*input_tensor(0, i, j-1, 0)
                                            +k2_xy_20*input_tensor(0, i+1, j-1, 1)
                                            +k2_xy_21*input_tensor(0, i+1, j, 1)
                                            +k2_xy_22*input_tensor(0, i, j, 1)
                                            +k2_xy_23*input_tensor(0, i, j-1, 1)
                                            ) *coef_2

                                            -(
                                            k2_yx_20*input_tensor(0, i+1, j-1, 0)
                                            +k2_yx_21*input_tensor(0, i+1, j, 0)
                                            +k2_yx_22*input_tensor(0, i, j, 0)
                                            +k2_yx_23*input_tensor(0, i, j-1, 0)
                                            +k2_yy_20*input_tensor(0, i+1, j-1, 1)
                                            +k2_yy_21*input_tensor(0, i+1, j, 1)
                                //        +k2_yy_22*input_tensor(0, i, j, 1)
                                            +k2_yy_23*input_tensor(0, i, j-1, 1)
                                            ) *coef_2);

            grad_weights_tensor(0,i,j,0) += grad_tensor(0,i,j,0) *
                                           ((
                                            k1_xx_30*input_tensor(0, i+1, j, 0)
                                            +k1_xx_31*input_tensor(0, i+1, j+1, 0)
                                            +k1_xx_32*input_tensor(0, i, j+1, 0)
                                //        +k1_xx_33*input_tensor(0, i, j, 0)
                                            +k1_xy_30*input_tensor(0, i+1, j, 1)
                                            +k1_xy_31*input_tensor(0, i+1, j+1, 1)
                                            +k1_xy_32*input_tensor(0, i, j+1, 1)
                                            +k1_xy_33*input_tensor(0, i, j, 1)
                                            ) *coef_1

                                            +(
                                            k1_yx_30*input_tensor(0, i+1, j, 0)
                                            +k1_yx_31*input_tensor(0, i+1, j+1, 0)
                                            +k1_yx_32*input_tensor(0, i, j+1, 0)
                                            +k1_yx_33*input_tensor(0, i, j, 0)
                                            +k1_yy_30*input_tensor(0, i+1, j, 1)
                                            +k1_yy_31*input_tensor(0, i+1, j+1, 1)
                                            +k1_yy_32*input_tensor(0, i, j+1, 1)
                                //        +k1_yy_33*input_tensor(0, i, j, 1)
                                            ) *coef_1

                                            -(
                                            k2_xx_30*input_tensor(0, i+1, j, 0)
                                            +k2_xx_31*input_tensor(0, i+1, j+1, 0)
                                            +k2_xx_32*input_tensor(0, i, j+1, 0)
                                //        +k2_xx_33*input_tensor(0, i, j, 0)
                                            +k2_xy_30*input_tensor(0, i+1, j, 1)
                                            +k2_xy_31*input_tensor(0, i+1, j+1, 1)
                                            +k2_xy_32*input_tensor(0, i, j+1, 1)
                                            +k2_xy_33*input_tensor(0, i, j, 1)
                                            ) *coef_2

                                            -(
                                            k2_yx_30*input_tensor(0, i+1, j, 0)
                                            +k2_yx_31*input_tensor(0, i+1, j+1, 0)
                                            +k2_yx_32*input_tensor(0, i, j+1, 0)
                                            +k2_yx_33*input_tensor(0, i, j, 0)
                                            +k2_yy_30*input_tensor(0, i+1, j, 1)
                                            +k2_yy_31*input_tensor(0, i+1, j+1, 1)
                                            +k2_yy_32*input_tensor(0, i, j+1, 1)
                                //        +k2_yy_33*input_tensor(0, i, j, 1)
                                            ) *coef_2);
      }
    }

    for (int i = 1; i < input_shape.dim_size(1)-1; i++) {
      for (int j = 1; j < input_shape.dim_size(2)-1; j++) {
//      grad_tensor(0,i-1,j-1,0) *
            grad_input_tensor(0,i-1,j-1,0) += grad_tensor(0,i-1,j-1,0) * (
                                            +k1_xy_13 * weights_tensor(0, i-1, j-1, 0)*coef_1
                                            +k1_yy_13* weights_tensor(0, i-1, j-1, 0)*coef_1
                                            +k2_xy_13* (1-weights_tensor(0, i-1, j-1, 0))*coef_2
                                            +k2_yy_13* (1-weights_tensor(0, i-1, j-1, 0))*coef_2);
            grad_input_tensor(0,i-1,j,0) += grad_tensor(0,i-1,j,0) * (
                                            k1_xy_12* weights_tensor(0, i-1, j-1, 0)*coef_1
                                            +k1_yy_12* weights_tensor(0, i-1, j-1, 0)*coef_1
                                            +k2_xy_12* (1-weights_tensor(0, i-1, j-1, 0))*coef_2
                                            +k2_yy_12* (1-weights_tensor(0, i-1, j-1, 0))*coef_2
                                            +k1_xy_03* weights_tensor(0, i-1, j, 0)*coef_1
                                            +k1_yy_03* weights_tensor(0, i-1, j, 0)*coef_1
                                            +k2_xy_03* (1-weights_tensor(0, i-1, j, 0))*coef_2
                                            +k2_yy_03* (1-weights_tensor(0, i-1, j, 0))*coef_2);
            grad_input_tensor(0,i-1,j+1,0) += grad_tensor(0,i-1,j+1,0) * (
                                            k1_xy_02 * weights_tensor(0, i-1, j, 0)*coef_1
                                            +k1_yy_02* weights_tensor(0, i-1, j, 0)*coef_1
                                            +k2_xy_02* (1-weights_tensor(0, i-1, j, 0))*coef_2
                                            +k2_yy_02* (1-weights_tensor(0, i-1, j, 0))*coef_2);

            grad_input_tensor(0,i,j-1,0) += grad_tensor(0,i,j-1,0) * (
                                            k1_xy_10* weights_tensor(0, i-1, j-1, 0)*coef_1
                                            +k1_yy_10* weights_tensor(0, i-1, j-1, 0)*coef_1
                                            +k2_xy_10* (1-weights_tensor(0, i-1, j-1, 0))*coef_2
                                            +k2_yy_10* (1-weights_tensor(0, i-1, j-1, 0))*coef_2
                                            +k1_xy_23* weights_tensor(0, i, j-1, 0)*coef_1
                                            +k1_yy_23* weights_tensor(0, i, j-1, 0)*coef_1
                                            +k2_xy_23* (1-weights_tensor(0, i, j-1, 0))*coef_2
                                            +k2_yy_23* (1-weights_tensor(0, i, j-1, 0))*coef_2);
            grad_input_tensor(0,i,j+1,0) += grad_tensor(0,i,j+1,0) * (
                                            k1_xy_01* weights_tensor(0, i-1, j, 0)*coef_1
                                            +k1_yy_01* weights_tensor(0, i-1, j, 0)*coef_1
                                            +k2_xy_01* (1-weights_tensor(0, i-1, j, 0))*coef_2
                                            +k2_yy_01* (1-weights_tensor(0, i-1, j, 0))*coef_2
                                            +k1_xy_32* weights_tensor(0, i, j, 0)*coef_1
                                            +k1_yy_32* weights_tensor(0, i, j, 0)*coef_1
                                            +k2_xy_32* (1-weights_tensor(0, i, j, 0))*coef_2
                                            +k2_yy_32* (1-weights_tensor(0, i, j, 0))*coef_2);

            grad_input_tensor(0,i+1,j-1,0) += grad_tensor(0,i+1,j-1,0) * (
                                            k1_xy_20 * weights_tensor(0, i+1, j-1, 0)*coef_1
                                            +k1_yy_20* weights_tensor(0, i+1, j-1, 0)*coef_1
                                            +k2_xy_20* (1-weights_tensor(0, i+1, j-1, 0))*coef_2
                                            +k2_yy_20* (1-weights_tensor(0, i+1, j-1, 0))*coef_2);
            grad_input_tensor(0,i+1,j,0) += grad_tensor(0,i+1,j,0) * (
                                            k1_xy_21* weights_tensor(0, i, j-1, 0)*coef_1
                                            +k1_yy_21* weights_tensor(0, i, j-1, 0)*coef_1
                                            +k2_xy_21* (1-weights_tensor(0, i, j-1, 0))*coef_2
                                            +k2_yy_21* (1-weights_tensor(0, i, j-1, 0))*coef_2
                                            +k1_xy_30* weights_tensor(0, i, j, 0)*coef_1
                                            +k1_yy_30* weights_tensor(0, i, j, 0)*coef_1
                                            +k2_xy_30* (1-weights_tensor(0, i, j, 0))*coef_2
                                            +k2_yy_30* (1-weights_tensor(0, i, j, 0))*coef_2);
            grad_input_tensor(0,i+1,j+1,0) += grad_tensor(0,i+1,j+1,0) * (
                                            k1_xy_31 * weights_tensor(0, i, j, 0)*coef_1
                                            +k1_yy_31* weights_tensor(0, i, j, 0)*coef_1
                                            +k2_xy_31* (1-weights_tensor(0, i, j, 0))*coef_2
                                            +k2_yy_31* (1-weights_tensor(0, i, j, 0))*coef_2);
      }
    }


    auto coef_1_grad_E =   1 / (16. * (1-mu_1*mu_1));
    auto coef_2_grad_E =   1 / (16. * (1-mu_2*mu_2));

    for (int i = 1; i < input_shape.dim_size(1)-1; i++) {
      for (int j = 1; j < input_shape.dim_size(2)-1; j++) {
            grad_rho_tensor(0) +=
                                +(
                                k1_xx_10*input_tensor(0, i, j-1, 0)
                    //        +k1_xx_11*input_tensor(0, i, j, 0)
                                +k1_xx_12*input_tensor(0, i-1, j, 0)
                                +k1_xx_13*input_tensor(0, i-1, j-1, 0)
                                +k1_xy_10*input_tensor(0, i, j-1, 1)
                                +k1_xy_11*input_tensor(0, i, j, 1)
                                +k1_xy_12*input_tensor(0, i-1, j, 1)
                                +k1_xy_13*input_tensor(0, i-1, j-1, 1)
                                ) * weights_tensor(0, i-1, j-1, 0)*coef_1_grad_E
                                +(
                                k1_yx_10*input_tensor(0, i, j-1, 0)
                                +k1_yx_11*input_tensor(0, i, j, 0)
                                +k1_yx_12*input_tensor(0, i-1, j, 0)
                                +k1_yx_13*input_tensor(0, i-1, j-1, 0)
                                +k1_yy_10*input_tensor(0, i, j-1, 1)
                    //        +k1_yy_11*input_tensor(0, i, j, 1)
                                +k1_yy_12*input_tensor(0, i-1, j, 1)
                                +k1_yy_13*input_tensor(0, i-1, j-1, 1)
                                 ) * weights_tensor(0, i-1, j-1, 0)*coef_1_grad_E

                                +(
                    //        k1_xx_00*input_tensor(0, i, j, 0)
                                +k1_xx_01*input_tensor(0, i, j+1, 0)
                                +k1_xx_02*input_tensor(0, i-1, j+1, 0)
                                +k1_xx_03*input_tensor(0, i-1, j, 0)
                                +k1_xy_00*input_tensor(0, i, j, 1)
                                +k1_xy_01*input_tensor(0, i, j+1, 1)
                                +k1_xy_02*input_tensor(0, i-1, j+1, 1)
                                +k1_xy_03*input_tensor(0, i-1, j, 1)
                                ) * weights_tensor(0, i-1, j, 0)*coef_1_grad_E
                                +(
                                k1_yx_00*input_tensor(0, i, j, 0)
                                +k1_yx_01*input_tensor(0, i, j+1, 0)
                                +k1_yx_02*input_tensor(0, i-1, j+1, 0)
                                +k1_yx_03*input_tensor(0, i-1, j, 0)
                    //        +k1_yy_00*input_tensor(0, i, j, 1)
                                +k1_yy_01*input_tensor(0, i, j+1, 1)
                                +k1_yy_02*input_tensor(0, i-1, j+1, 1)
                                +k1_yy_03*input_tensor(0, i-1, j, 1)
                                ) * weights_tensor(0, i-1, j, 0)*coef_1_grad_E

                                +(
                                k1_xx_20*input_tensor(0, i+1, j-1, 0)
                                +k1_xx_21*input_tensor(0, i+1, j, 0)
                    //        +k1_xx_22*input_tensor(0, i, j, 0)
                                +k1_xx_23*input_tensor(0, i, j-1, 0)
                                +k1_xy_20*input_tensor(0, i+1, j-1, 1)
                                +k1_xy_21*input_tensor(0, i+1, j, 1)
                                +k1_xy_22*input_tensor(0, i, j, 1)
                                +k1_xy_23*input_tensor(0, i, j-1, 1)
                                ) * weights_tensor(0, i, j-1, 0)*coef_1_grad_E
                                +(
                                k1_yx_20*input_tensor(0, i+1, j-1, 0)
                                +k1_yx_21*input_tensor(0, i+1, j, 0)
                                +k1_yx_22*input_tensor(0, i, j, 0)
                                +k1_yx_23*input_tensor(0, i, j-1, 0)
                                +k1_yy_20*input_tensor(0, i+1, j-1, 1)
                                +k1_yy_21*input_tensor(0, i+1, j, 1)
                    //        +k1_yy_22*input_tensor(0, i, j, 1)
                                +k1_yy_23*input_tensor(0, i, j-1, 1)
                                ) * weights_tensor(0, i, j-1, 0)*coef_1_grad_E

                               +(
                                k1_xx_30*input_tensor(0, i+1, j, 0)
                                +k1_xx_31*input_tensor(0, i+1, j+1, 0)
                                +k1_xx_32*input_tensor(0, i, j+1, 0)
                    //        +k1_xx_33*input_tensor(0, i, j, 0)
                                +k1_xy_30*input_tensor(0, i+1, j, 1)
                                +k1_xy_31*input_tensor(0, i+1, j+1, 1)
                                +k1_xy_32*input_tensor(0, i, j+1, 1)
                                +k1_xy_33*input_tensor(0, i, j, 1)
                                ) * weights_tensor(0, i, j, 0)*coef_1_grad_E
                                +(
                                k1_yx_30*input_tensor(0, i+1, j, 0)
                                +k1_yx_31*input_tensor(0, i+1, j+1, 0)
                                +k1_yx_32*input_tensor(0, i, j+1, 0)
                                +k1_yx_33*input_tensor(0, i, j, 0)
                                +k1_yy_30*input_tensor(0, i+1, j, 1)
                                +k1_yy_31*input_tensor(0, i+1, j+1, 1)
                                +k1_yy_32*input_tensor(0, i, j+1, 1)
                    //        +k1_yy_33*input_tensor(0, i, j, 1)
                                ) * weights_tensor(0, i, j, 0)*coef_1_grad_E;


            grad_rho_tensor(2) +=
                                +(
                                k2_xx_10*input_tensor(0, i, j-1, 0)
                    //        +k2_xx_11*input_tensor(0, i, j, 0)
                                +k2_xx_12*input_tensor(0, i-1, j, 0)
                                +k2_xx_13*input_tensor(0, i-1, j-1, 0)
                                +k2_xy_10*input_tensor(0, i, j-1, 1)
                                +k2_xy_11*input_tensor(0, i, j, 1)
                                +k2_xy_12*input_tensor(0, i-1, j, 1)
                                +k2_xy_13*input_tensor(0, i-1, j-1, 1)
                                ) * (1-weights_tensor(0, i-1, j-1, 0))*coef_2_grad_E
                                +(
                                k2_yx_10*input_tensor(0, i, j-1, 0)
                                +k2_yx_11*input_tensor(0, i, j, 0)
                                +k2_yx_12*input_tensor(0, i-1, j, 0)
                                +k2_yx_13*input_tensor(0, i-1, j-1, 0)
                                +k2_yy_10*input_tensor(0, i, j-1, 1)
                    //        +k2_yy_11*input_tensor(0, i, j, 1)
                                +k2_yy_12*input_tensor(0, i-1, j, 1)
                                +k2_yy_13*input_tensor(0, i-1, j-1, 1)
                                ) * (1-weights_tensor(0, i-1, j-1, 0))*coef_2_grad_E

                                +(
                    //        k2_xx_00*input_tensor(0, i, j, 0)
                                +k2_xx_01*input_tensor(0, i, j+1, 0)
                                +k2_xx_02*input_tensor(0, i-1, j+1, 0)
                                +k2_xx_03*input_tensor(0, i-1, j, 0)
                                +k2_xy_00*input_tensor(0, i, j, 1)
                                +k2_xy_01*input_tensor(0, i, j+1, 1)
                                +k2_xy_02*input_tensor(0, i-1, j+1, 1)
                                +k2_xy_03*input_tensor(0, i-1, j, 1)
                                ) * (1-weights_tensor(0, i-1, j, 0))*coef_2_grad_E
                                +(
                                k2_yx_00*input_tensor(0, i, j, 0)
                                +k2_yx_01*input_tensor(0, i, j+1, 0)
                                +k2_yx_02*input_tensor(0, i-1, j+1, 0)
                                +k2_yx_03*input_tensor(0, i-1, j, 0)
                    //        +k2_yy_00*input_tensor(0, i, j, 1)
                                +k2_yy_01*input_tensor(0, i, j+1, 1)
                                +k2_yy_02*input_tensor(0, i-1, j+1, 1)
                                +k2_yy_03*input_tensor(0, i-1, j, 1)
                                ) * (1-weights_tensor(0, i-1, j, 0))*coef_2_grad_E

                                 +(
                                k2_xx_20*input_tensor(0, i+1, j-1, 0)
                                +k2_xx_21*input_tensor(0, i+1, j, 0)
                    //        +k2_xx_22*input_tensor(0, i, j, 0)
                                +k2_xx_23*input_tensor(0, i, j-1, 0)
                                +k2_xy_20*input_tensor(0, i+1, j-1, 1)
                                +k2_xy_21*input_tensor(0, i+1, j, 1)
                                +k2_xy_22*input_tensor(0, i, j, 1)
                                +k2_xy_23*input_tensor(0, i, j-1, 1)
                                ) * (1-weights_tensor(0, i, j-1, 0))*coef_2_grad_E
                                +(
                                k2_yx_20*input_tensor(0, i+1, j-1, 0)
                                +k2_yx_21*input_tensor(0, i+1, j, 0)
                                +k2_yx_22*input_tensor(0, i, j, 0)
                                +k2_yx_23*input_tensor(0, i, j-1, 0)
                                +k2_yy_20*input_tensor(0, i+1, j-1, 1)
                                +k2_yy_21*input_tensor(0, i+1, j, 1)
                    //        +k2_yy_22*input_tensor(0, i, j, 1)
                                +k2_yy_23*input_tensor(0, i, j-1, 1)
                                ) * (1-weights_tensor(0, i, j-1, 0))*coef_2_grad_E

                                +(
                                k2_xx_30*input_tensor(0, i+1, j, 0)
                                +k2_xx_31*input_tensor(0, i+1, j+1, 0)
                                +k2_xx_32*input_tensor(0, i, j+1, 0)
                    //        +k2_xx_33*input_tensor(0, i, j, 0)
                                +k2_xy_30*input_tensor(0, i+1, j, 1)
                                +k2_xy_31*input_tensor(0, i+1, j+1, 1)
                                +k2_xy_32*input_tensor(0, i, j+1, 1)
                                +k2_xy_33*input_tensor(0, i, j, 1)
                                ) * (1-weights_tensor(0, i, j, 0))*coef_2_grad_E
                                +(
                                k2_yx_30*input_tensor(0, i+1, j, 0)
                                +k2_yx_31*input_tensor(0, i+1, j+1, 0)
                                +k2_yx_32*input_tensor(0, i, j+1, 0)
                                +k2_yx_33*input_tensor(0, i, j, 0)
                                +k2_yy_30*input_tensor(0, i+1, j, 1)
                                +k2_yy_31*input_tensor(0, i+1, j+1, 1)
                                +k2_yy_32*input_tensor(0, i, j+1, 1)
                    //        +k2_yy_33*input_tensor(0, i, j, 1)
                                ) * (1-weights_tensor(0, i, j, 0))*coef_2_grad_E;


            grad_rho_tensor(1) +=
                            +(
                            k1_xx_10_coef_grad_mu*input_tensor(0, i, j-1, 0)
                //        +k1_xx_11*input_tensor(0, i, j, 0)
                            +k1_xx_12_coef_grad_mu*input_tensor(0, i-1, j, 0)
                            +k1_xx_13_coef_grad_mu*input_tensor(0, i-1, j-1, 0)
                            +k1_xy_10_coef_grad_mu*input_tensor(0, i, j-1, 1)
                            +k1_xy_11_coef_grad_mu*input_tensor(0, i, j, 1)
                            +k1_xy_12_coef_grad_mu*input_tensor(0, i-1, j, 1)
                            +k1_xy_13_coef_grad_mu*input_tensor(0, i-1, j-1, 1)
                            ) * weights_tensor(0, i-1, j-1, 0)
                            +(
                            k1_yx_10_coef_grad_mu*input_tensor(0, i, j-1, 0)
                            +k1_yx_11_coef_grad_mu*input_tensor(0, i, j, 0)
                            +k1_yx_12_coef_grad_mu*input_tensor(0, i-1, j, 0)
                            +k1_yx_13_coef_grad_mu*input_tensor(0, i-1, j-1, 0)
                            +k1_yy_10_coef_grad_mu*input_tensor(0, i, j-1, 1)
                //        +k1_yy_11*input_tensor(0, i, j, 1)
                            +k1_yy_12_coef_grad_mu*input_tensor(0, i-1, j, 1)
                            +k1_yy_13_coef_grad_mu*input_tensor(0, i-1, j-1, 1)
                             ) * weights_tensor(0, i-1, j-1, 0)

                            +(
                //        k1_xx_00*input_tensor(0, i, j, 0)
                            +k1_xx_01_coef_grad_mu*input_tensor(0, i, j+1, 0)
                            +k1_xx_02_coef_grad_mu*input_tensor(0, i-1, j+1, 0)
                            +k1_xx_03_coef_grad_mu*input_tensor(0, i-1, j, 0)
                            +k1_xy_00_coef_grad_mu*input_tensor(0, i, j, 1)
                            +k1_xy_01_coef_grad_mu*input_tensor(0, i, j+1, 1)
                            +k1_xy_02_coef_grad_mu*input_tensor(0, i-1, j+1, 1)
                            +k1_xy_03_coef_grad_mu*input_tensor(0, i-1, j, 1)
                            ) * weights_tensor(0, i-1, j, 0)

                            +(
                            k1_yx_00_coef_grad_mu*input_tensor(0, i, j, 0)
                            +k1_yx_01_coef_grad_mu*input_tensor(0, i, j+1, 0)
                            +k1_yx_02_coef_grad_mu*input_tensor(0, i-1, j+1, 0)
                            +k1_yx_03_coef_grad_mu*input_tensor(0, i-1, j, 0)
                //        +k1_yy_00*input_tensor(0, i, j, 1)
                            +k1_yy_01_coef_grad_mu*input_tensor(0, i, j+1, 1)
                            +k1_yy_02_coef_grad_mu*input_tensor(0, i-1, j+1, 1)
                            +k1_yy_03_coef_grad_mu*input_tensor(0, i-1, j, 1)
                            ) * weights_tensor(0, i-1, j, 0)

                            +(
                            k1_xx_20_coef_grad_mu*input_tensor(0, i+1, j-1, 0)
                            +k1_xx_21_coef_grad_mu*input_tensor(0, i+1, j, 0)
                //        +k1_xx_22*input_tensor(0, i, j, 0)
                            +k1_xx_23_coef_grad_mu*input_tensor(0, i, j-1, 0)
                            +k1_xy_20_coef_grad_mu*input_tensor(0, i+1, j-1, 1)
                            +k1_xy_21_coef_grad_mu*input_tensor(0, i+1, j, 1)
                            +k1_xy_22_coef_grad_mu*input_tensor(0, i, j, 1)
                            +k1_xy_23_coef_grad_mu*input_tensor(0, i, j-1, 1)
                            ) * weights_tensor(0, i, j-1, 0)
                            +(
                            k1_yx_20_coef_grad_mu*input_tensor(0, i+1, j-1, 0)
                            +k1_yx_21_coef_grad_mu*input_tensor(0, i+1, j, 0)
                            +k1_yx_22_coef_grad_mu*input_tensor(0, i, j, 0)
                            +k1_yx_23_coef_grad_mu*input_tensor(0, i, j-1, 0)
                            +k1_yy_20_coef_grad_mu*input_tensor(0, i+1, j-1, 1)
                            +k1_yy_21_coef_grad_mu*input_tensor(0, i+1, j, 1)
                //        +k1_yy_22*input_tensor(0, i, j, 1)
                            +k1_yy_23_coef_grad_mu*input_tensor(0, i, j-1, 1)
                            ) * weights_tensor(0, i, j-1, 0)

                           +(
                            k1_xx_30_coef_grad_mu*input_tensor(0, i+1, j, 0)
                            +k1_xx_31_coef_grad_mu*input_tensor(0, i+1, j+1, 0)
                            +k1_xx_32_coef_grad_mu*input_tensor(0, i, j+1, 0)
                //        +k1_xx_33*input_tensor(0, i, j, 0)
                            +k1_xy_30_coef_grad_mu*input_tensor(0, i+1, j, 1)
                            +k1_xy_31_coef_grad_mu*input_tensor(0, i+1, j+1, 1)
                            +k1_xy_32_coef_grad_mu*input_tensor(0, i, j+1, 1)
                            +k1_xy_33_coef_grad_mu*input_tensor(0, i, j, 1)
                            ) * weights_tensor(0, i, j, 0)
                            +(
                            k1_yx_30_coef_grad_mu*input_tensor(0, i+1, j, 0)
                            +k1_yx_31_coef_grad_mu*input_tensor(0, i+1, j+1, 0)
                            +k1_yx_32_coef_grad_mu*input_tensor(0, i, j+1, 0)
                            +k1_yx_33_coef_grad_mu*input_tensor(0, i, j, 0)
                            +k1_yy_30_coef_grad_mu*input_tensor(0, i+1, j, 1)
                            +k1_yy_31_coef_grad_mu*input_tensor(0, i+1, j+1, 1)
                            +k1_yy_32_coef_grad_mu*input_tensor(0, i, j+1, 1)
                //        +k1_yy_33*input_tensor(0, i, j, 1)
                            ) * weights_tensor(0, i, j, 0);

            grad_rho_tensor(3) +=
                            +(
                            k2_xx_10_coef_grad_mu*input_tensor(0, i, j-1, 0)
                //        +k2_xx_11*input_tensor(0, i, j, 0)
                            +k2_xx_12_coef_grad_mu*input_tensor(0, i-1, j, 0)
                            +k2_xx_13_coef_grad_mu*input_tensor(0, i-1, j-1, 0)
                            +k2_xy_10_coef_grad_mu*input_tensor(0, i, j-1, 1)
                            +k2_xy_11_coef_grad_mu*input_tensor(0, i, j, 1)
                            +k2_xy_12_coef_grad_mu*input_tensor(0, i-1, j, 1)
                            +k2_xy_13_coef_grad_mu*input_tensor(0, i-1, j-1, 1)
                            ) * (1-weights_tensor(0, i-1, j-1, 0))
                            +(
                            k2_yx_10_coef_grad_mu*input_tensor(0, i, j-1, 0)
                            +k2_yx_11_coef_grad_mu*input_tensor(0, i, j, 0)
                            +k2_yx_12_coef_grad_mu*input_tensor(0, i-1, j, 0)
                            +k2_yx_13_coef_grad_mu*input_tensor(0, i-1, j-1, 0)
                            +k2_yy_10_coef_grad_mu*input_tensor(0, i, j-1, 1)
                //        +k2_yy_11*input_tensor(0, i, j, 1)
                            +k2_yy_12_coef_grad_mu*input_tensor(0, i-1, j, 1)
                            +k2_yy_13_coef_grad_mu*input_tensor(0, i-1, j-1, 1)
                            ) * (1-weights_tensor(0, i-1, j-1, 0))

                            +(
                //        k2_xx_00*input_tensor(0, i, j, 0)
                            +k2_xx_01_coef_grad_mu*input_tensor(0, i, j+1, 0)
                            +k2_xx_02_coef_grad_mu*input_tensor(0, i-1, j+1, 0)
                            +k2_xx_03_coef_grad_mu*input_tensor(0, i-1, j, 0)
                            +k2_xy_00_coef_grad_mu*input_tensor(0, i, j, 1)
                            +k2_xy_01_coef_grad_mu*input_tensor(0, i, j+1, 1)
                            +k2_xy_02_coef_grad_mu*input_tensor(0, i-1, j+1, 1)
                            +k2_xy_03_coef_grad_mu*input_tensor(0, i-1, j, 1)
                            ) * (1-weights_tensor(0, i-1, j, 0))
                            +(
                            k2_yx_00_coef_grad_mu*input_tensor(0, i, j, 0)
                            +k2_yx_01_coef_grad_mu*input_tensor(0, i, j+1, 0)
                            +k2_yx_02_coef_grad_mu*input_tensor(0, i-1, j+1, 0)
                            +k2_yx_03_coef_grad_mu*input_tensor(0, i-1, j, 0)
                //        +k2_yy_00*input_tensor(0, i, j, 1)
                            +k2_yy_01_coef_grad_mu*input_tensor(0, i, j+1, 1)
                            +k2_yy_02_coef_grad_mu*input_tensor(0, i-1, j+1, 1)
                            +k2_yy_03_coef_grad_mu*input_tensor(0, i-1, j, 1)
                            ) * (1-weights_tensor(0, i-1, j, 0))

                             +(
                            k2_xx_20_coef_grad_mu*input_tensor(0, i+1, j-1, 0)
                            +k2_xx_21_coef_grad_mu*input_tensor(0, i+1, j, 0)
                //        +k2_xx_22*input_tensor(0, i, j, 0)
                            +k2_xx_23_coef_grad_mu*input_tensor(0, i, j-1, 0)
                            +k2_xy_20_coef_grad_mu*input_tensor(0, i+1, j-1, 1)
                            +k2_xy_21_coef_grad_mu*input_tensor(0, i+1, j, 1)
                            +k2_xy_22_coef_grad_mu*input_tensor(0, i, j, 1)
                            +k2_xy_23_coef_grad_mu*input_tensor(0, i, j-1, 1)
                            ) * (1-weights_tensor(0, i, j-1, 0))
                            +(
                            k2_yx_20_coef_grad_mu*input_tensor(0, i+1, j-1, 0)
                            +k2_yx_21_coef_grad_mu*input_tensor(0, i+1, j, 0)
                            +k2_yx_22_coef_grad_mu*input_tensor(0, i, j, 0)
                            +k2_yx_23_coef_grad_mu*input_tensor(0, i, j-1, 0)
                            +k2_yy_20_coef_grad_mu*input_tensor(0, i+1, j-1, 1)
                            +k2_yy_21_coef_grad_mu*input_tensor(0, i+1, j, 1)
                //        +k2_yy_22*input_tensor(0, i, j, 1)
                            +k2_yy_23_coef_grad_mu*input_tensor(0, i, j-1, 1)
                            ) * (1-weights_tensor(0, i, j-1, 0))

                            +(
                            k2_xx_30_coef_grad_mu*input_tensor(0, i+1, j, 0)
                            +k2_xx_31_coef_grad_mu*input_tensor(0, i+1, j+1, 0)
                            +k2_xx_32_coef_grad_mu*input_tensor(0, i, j+1, 0)
                //        +k2_xx_33*input_tensor(0, i, j, 0)
                            +k2_xy_30_coef_grad_mu*input_tensor(0, i+1, j, 1)
                            +k2_xy_31_coef_grad_mu*input_tensor(0, i+1, j+1, 1)
                            +k2_xy_32_coef_grad_mu*input_tensor(0, i, j+1, 1)
                            +k2_xy_33_coef_grad_mu*input_tensor(0, i, j, 1)
                            ) * (1-weights_tensor(0, i, j, 0))
                            +(
                            k2_yx_30_coef_grad_mu*input_tensor(0, i+1, j, 0)
                            +k2_yx_31_coef_grad_mu*input_tensor(0, i+1, j+1, 0)
                            +k2_yx_32_coef_grad_mu*input_tensor(0, i, j+1, 0)
                            +k2_yx_33_coef_grad_mu*input_tensor(0, i, j, 0)
                            +k2_yy_30_coef_grad_mu*input_tensor(0, i+1, j, 1)
                            +k2_yy_31_coef_grad_mu*input_tensor(0, i+1, j+1, 1)
                            +k2_yy_32_coef_grad_mu*input_tensor(0, i, j+1, 1)
                //        +k2_yy_33*input_tensor(0, i, j, 1)
                            ) * (1-weights_tensor(0, i, j, 0));

      }
    }

  }
};

REGISTER_KERNEL_BUILDER(Name("MaskconvElastGrad").Device(DEVICE_CPU), MaskconvElastGradOp);
