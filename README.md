# stochastic-gradient-descent
Implement stochastic gradient descent with Pegasos algorithm and softplus function. Then, evaluate its performance on MNIST-13 dataset

## Code Overview
### sgd.py
This includes stochastic gradient descent with Pegasos algorithm and softplus function.

### Pegasos algorithm
The Pegasos algorithm is implemented in the following process:
1. Extract feature $X$ and label $y$ from data file.
2. Standardized feature X and augment feature $X$ by 1 to account for the bias term.
3. Reclassify $y$ to ${-1,1}$ for $y∈{1,3}$
4. Initiate the weights, $w$ as random number and set $\lambda=0.001$
5. Split dataset into 2 groups based on the label $y$.
6. Initiate termination condition: $ktot=100 \times n$. This is the number of iterations we will run for the steps below.
7. For each iteration, $t$, randomly select $k%$ from each group and combine the dataset to allow balanced training. This dataset is used for the minibatch update.
8. For the dataset that is selected in step 7, calculate their gradient:
   - Set learning rate $\eta=\frac{1}{(\lambda(t+1))}$ where $t$ is the $t^{th}$  iteration of the gradient update
   - Identify data entry that has been misclassified using $y_i (w^T X_i )<1$
   - For those data points that has been misclassified, calculate $w_{1/2}=(1-\eta\lambda)w+ \frac{\eta}{k} y_+X_+$ where k is the total number of the minibatch and $y_+X_+$  are the misclassified points
   - The weight, $w$ is then calculated.
9. The final weight, w is the weight that is calculated at the termination condition: $w_{final}=w_{ktot}$
10. The loss function and primal objective are defined as the hinge loss.

The above process is repeated for
$k=0.05%,1%,10%,50%,100%$

And run each $k$ for 5 times.

### Softplus function
The process for softplus is the same as Pegasos algorithm above except for step 8 to 10. 
- From step 9, we will use the derived gradient above to update w: $w = w − \eta \Delta L(w)$, where \eta is the learning rate and it is set as
  $$\eta = \lambda = 0.001$$
  And $\alpha$ from the softplus function is set as $\alpha = 0.05$
  The final weight, $w$ is the final $w$ that is converged to at the termination.

### Pegasos.ipynb
Performance of stochastic gradient descent with Pegasos algorithm on MNIST-13 dataset

### SoftPlus.ipynb
Performance of stochastic gradient descent with softplus function on MNIST-13 dataset
