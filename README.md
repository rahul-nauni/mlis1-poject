# Project for MLiS part I

## Aim 
To implement regression in python without using machine learning libraries i.e., scikit-learn, keras, tensorflow, pytorch, etc.


## Model

$$ \hat{y} =  \theta_0 + \theta_1x  $$


### Loss Function

**MSE Loss**

$$ L(y, \hat{y}) = 1/N \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 $$

$$ L(y,x,\theta) = 1/N \sum_{i=1}^{N} (y_i - (\theta_0 + \theta_1x_i))^2 $$

**L2 Loss**


$$ L(\theta) = 1/2N \sum_{i=1}^{N} (y_i  - \hat{y}_i)^2 + λ/2N \sum_{j=1}^{P}{\theta_j}^2 $$


### Derivative computation

Calculation of partial derivatives for every parameter:

$$ ∂L / ∂\theta_0 =  -2/N \sum_{i=1}^{N} (y_i - (\theta_0 + \theta_1x_i)) $$

$$ ∂L / ∂\theta_1 =  -2/N \sum_{i=1}^{N} x_i(y_i - (\theta_0 + \theta_1x_i)) $$

**Derivative of L2 loss**


$$ ∂L(\theta) / ∂θ_0 = -1/N \sum_{i=1}^{N} (y_i - (\theta_0 + \theta_1x_i)) + λ/N \sum_{j=1}^{P} \theta_j $$

### Fitted line using gradient descent

![image](https://user-images.githubusercontent.com/116102220/213288946-2cd004c8-cbdb-45e1-b429-506678cd75ef.png)

### Visualizing loss vs theta over epochs

![image](https://user-images.githubusercontent.com/116102220/213289183-496b4ecf-d0ce-4ae4-b996-db4e09c8ec94.png)

