
1. convolution with special boundary treatment is equvlent
![u_gt](./data/heat_transfer/ground_truth_u.png)
![u_conv](./data/heat_transfer/jacobi_wx_it612.png)


2. feed forward Jacobi network is converging, but slow
![Jacobi_forward_convergence](./data/heat_transfer/jacobi_wx_convergence.png)

3. parameter estimation requires very deep network:

ground truth -> 16.0

400 layers -> 8.1

1500 layers -> 14.3

2000 layers -> 15.0

This is caused by the error in network prediction/ slow convergence v.s. network depth

4. Higher order of upsampling will improve the accuracy.
Down sample will casue error to accumulate if the number of pixels at one side is not even.

VMG accuracy:

2 level (50,20), 11.68%

3 level (50,20), 10.97%

4 level (50,20), 10.64%

4 level (50,50), 10.61%


