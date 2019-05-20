/// \file inner_product.cc
/// \author David Stutz
/// \brief Implementation of a inner product (i.e. fully connected layer)
/// operation in Tensorflow.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// MUST BE IN CamelCase!!
REGISTER_OP("GetdmatElast")
  .Input("input: float")
  .Input("weights: float")
  .Input("rho: float")
  .Output("mask_conv: float");

/// \brief Implementation of an inner product operation.
/// \param context
/// \author David Stutz
class GetdmatElastOp : public OpKernel {
public:
  /// \brief Constructor.
  /// \param context
  explicit GetdmatElastOp(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  /// \brief Compute the inner product.
  /// \param context
  void Compute(OpKernelContext* context) override {
    
    // some checks to be sure ...
    DCHECK_EQ(3, context->num_inputs());
    
    // get the input tensor
//    const Tensor& input = context->input(0);
    
    // get the weight tensor
    const Tensor& weights = context->input(1);

    // get the heat conductivity
    const Tensor& rho = context->input(2);

    // check shapes of input and weights
//    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weights.shape();
    
    // check input is a standing vector
//    DCHECK_EQ(input_shape.dims(), 4);
    //DCHECK_EQ(input_shape.dim_size(1), input_shape.dim_size(2));
    
    // check weights is matrix of correct size
    DCHECK_EQ(weights_shape.dims(), 4);
    //DCHECK_EQ(input_shape.dim_size(2), weights_shape.dim_size(2));
    
    // create output shape
    TensorShape output_shape;
    output_shape.AddDim(weights_shape.dim_size(0));
    output_shape.AddDim(weights_shape.dim_size(1)+1);
    output_shape.AddDim(weights_shape.dim_size(2)+1);
    output_shape.AddDim(weights_shape.dim_size(3));

    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
//    auto input_tensor = input.tensor<float, 4>();
    auto weights_tensor = weights.tensor<float, 4>();
    auto rho_tensor = rho.tensor<float, 1>();
    auto output_tensor = output->tensor<float, 4>();

    auto E_1 = rho_tensor(0);
    auto mu_1 = rho_tensor(1);
    auto E_2 = rho_tensor(2);
    auto mu_2 = rho_tensor(3);

    auto coef_1 = E_1 / (16. * (1-mu_1*mu_1));
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
    // k1_yy, off-diagonal, PROBLEMATIC!!
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


    auto coef_2 = E_2 / (16. * (1-mu_2*mu_2));
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
//    std::cout<<"k2xx  "<< k2_xx_20<< " "<< k2_xx_31 <<std::endl;
//    std::cout<<"k2xx  "<< k2_xx_02<< " "<< k1_xx_13 <<std::endl;
//    std::cout<<"k2xx  "<< 4/3.*mu_2 - 4.<< " "<< k1_xx_13 <<std::endl;
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


    for (int i = 1; i < output->shape().dim_size(1)+1; i++) {
        for (int j = 1; j < output->shape().dim_size(2)+1; j++) {
            // from eq.11, mask_conv_expansion
            auto x_mat1 =
            (
        +k1_xx_22
            ) * weights_tensor(0, i, j-1, 0)

            +(
        +k1_xx_33
            ) * weights_tensor(0, i, j, 0)

            +(
        k1_xx_00
            ) * weights_tensor(0, i-1, j, 0)

            +(
        +k1_xx_11
            ) * weights_tensor(0, i-1, j-1, 0);

            auto y_mat1 =
            (
        +k1_yy_22
            ) * weights_tensor(0, i, j-1, 0)

            +(
        +k1_yy_33
            ) * weights_tensor(0, i, j, 0)

            +(
        +k1_yy_00
            ) * weights_tensor(0, i-1, j, 0)

            +(
        +k1_yy_11
             ) * weights_tensor(0, i-1, j-1, 0);

            auto x_mat2 =
            (
        +k2_xx_22
            ) * (1-weights_tensor(0, i, j-1, 0))

            +(
        +k2_xx_33
            ) * (1-weights_tensor(0, i, j, 0))

            +(
        k2_xx_00
            ) * (1-weights_tensor(0, i-1, j, 0))

            +(
        +k2_xx_11
            ) * (1-weights_tensor(0, i-1, j-1, 0));

            auto y_mat2 =
            (
        +k2_yy_22
            ) * (1-weights_tensor(0, i, j-1, 0))

            +(
        +k2_yy_33
            ) * (1-weights_tensor(0, i, j, 0))

            +(
        +k2_yy_00
            ) * (1-weights_tensor(0, i-1, j, 0))

            +(
        +k2_yy_11
            ) * (1-weights_tensor(0, i-1, j-1, 0));

            output_tensor(0, i-1, j-1, 0)  = x_mat1*coef_1 + x_mat2*coef_2;
            output_tensor(0, i-1, j-1, 1)  = y_mat1*coef_1 +y_mat2*coef_2;

        }
    }


  }
};

REGISTER_KERNEL_BUILDER(Name("GetdmatElast").Device(DEVICE_CPU), GetdmatElastOp);
