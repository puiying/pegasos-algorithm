# pegasos-algorithm
Implement Pegasos algorithm and evaluate its performance on MNIST-13 dataset

## Code Overview
The Pegasos algorithm is implemented in the following process:
1. Extract feature X and label y from data file.
2. Standardized feature X and augment feature X by 1 to account for the bias term.
3. Reclassify y to {-1,1} for y∈{1,3}
4. Initiate the weights, w as random number and set λ=0.001
5. Split dataset into 2 groups based on the label y.
6. Initiate termination condition: ktot=100×n. This is the number of iterations we will run for the steps below.
7. For each iteration, t, randomly select k% from each group and combine the dataset to allow balanced training. This dataset is used for the minibatch update.
8. For the dataset that is selected in step 7, calculate their gradient:
   - Set learning rate η=1/(λ(t+1)) where t is the t^th  iteration of the gradient update
   - Identify data entry that has been misclassified using y_i (w^T X_i )<1
   - For those data points that has been misclassified, calculate w_(1/2)=(1-ηλ)w+η/k y_+ X_+ where k is the total number of the minibatch and y_+ X_+  are the misclassified points
   - The weight, w is then calculated w=min⁡(1,(1/√λ)/‖w_(1/2) ‖ )×w_(1/2)
10. The final weight, w is the weight that is calculated at the termination condition w_final=w_ktot
11. The loss function and primal objective is defined as the hinge loss: L(w)=1/N ∑_(i=1:N)▒〖max⁡(0,y_i w^T X_i )+λ〗 ‖w‖^2

The above process is repeated for
k={0.05%,1%,10%,50%,100%}

And run each k for 5 times.
