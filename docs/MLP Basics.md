# Chapter 1
Bias is added to the weighted sum *before* passing it through the sigmoid function.

### 1. The Neuron Activation Equation
$$
a_0^{(1)} = \overset{\text{Sigmoid}}{\sigma} \left( w_{0,0} a_0^{(0)} + w_{0,1} a_1^{(0)} + \cdots + w_{0,n} a_n^{(0)} + \underset{\text{Bias}}{b_0} \right)
$$
We represent the above equation as a matrix. This is the standard notation used when studying about neural networks
### 2. The Matrix Representation
$$
\begin{bmatrix}
w_{0,0} & w_{0,1} & \cdots & w_{0,n} \\
w_{1,0} & w_{1,1} & \cdots & w_{1,n} \\
\vdots & \vdots & \ddots & \vdots \\
w_{k,0} & w_{k,1} & \cdots & w_{k,n}
\end{bmatrix}
\begin{bmatrix}
a_0^{(0)} \\
a_1^{(0)} \\
\vdots \\
a_n^{(0)}
\end{bmatrix}
$$
Here, we will get a column vector that will give weighted sum for each neuron in the next layer.
There are k neurons in the next layer for the above example. We will now add the bias (which is another column vector) and pass the column vector inside the sigmoid function. This will give the final column vector which represents activation of the next layer.
$$
\huge{\sigma} \left(
\begin{bmatrix}
w_{0,0} & w_{0,1} & \cdots & w_{0,n} \\
w_{1,0} & w_{1,1} & \cdots & w_{1,n} \\
\vdots & \vdots & \ddots & \vdots \\
w_{k,0} & w_{k,1} & \cdots & w_{k,n}
\end{bmatrix}
\begin{bmatrix}
a_0^{(0)} \\
a_1^{(0)} \\
\vdots \\
a_n^{(0)}
\end{bmatrix}
+
\begin{bmatrix}
b_0\\
b_1\\
\vdots\\
b_k\\
\end{bmatrix}
\right)

$$
Let the matrix representing the weights be W, the activation of the previous layer be a, and the bias vector be b.
So, we can represent moving from one layer to another using the equation:
$a^{1}=\sigma(W.a^{0}+b)$
($a^1$ layer has k neurons while $a^0$ layer has n neurons here).

Modern neural networks use ReLU (rectified linear unit) as it is easier to train and gives good results.
$ReLU(a)=max(0,a)$

# Chapter 2
## Cost function:
We need to be able to tell the computer how wrong it has predicted the output (digit in this case).
Suppose our last layer has 10 neurons. We feed the network a image of 3.
We expect the output to be: all neurons are zero except the one that represents 3.
The cost function is a way to showing that. We take the difference of the output and expected output for each neuron, square it, and add them. We find the average of this over all the training set and add it.
$$
\text{Cost} = \frac{1}{m} \sum_{i=1}^{m} \left(
\begin{bmatrix} a_0 \\ a_1 \\ a_2 \end{bmatrix}^{(i)} -
\begin{bmatrix} o_0 \\ o_1 \\ o_2 \end{bmatrix}^{(i)}
\right)^2
$$
(a matrix here is the output and the o matrix is the expected output. Cost is the sum of all these squares)

We need to find the average cost of the network over all the training data, and we need to minimize this cost.

|                | Input                  | Output             | Parameters                                 |
| -------------- | ---------------------- | ------------------ | ------------------------------------------ |
| Neural Network | 784 Pixel values       | 10 numbers         | 13k weights and biases                     |
| Cost Function  | 13k weights and biases | Single cost number | The training set of images (training data) |
 
 **We now have to find the minima of this cost function**

Gradient of a function gives the direction of steepest increase.
Therefore, the minimum will be in the opposite direction of that.
This vector (with 13k components in this case), will give the direction of steepest descent and will help us find the correct weights and biases for the neural network.

# Chapter 3 & 4

## Backpropogation 
is the algorithm used to decide how a single training example would like to shift the weights and biases. We add all these "shifts" and get something that resembles the negative gradient of the cost function
## Stochastic gradient descent 
is when we take small steps towards the minimum using small batches of training data instead of taking steps using the whole training data. This is computationally much easier to do, and we still reach the same minima just take a longer less accurate path there. 

## 4 main formulae for backpropogation
### The 4 Equations of Backpropagation

**1. Error in the Output Layer**
$$ \delta^L = \nabla_a C \odot \sigma'(z^L) $$
$\nabla_a C$ here is a column vector with $\frac{\partial C}{\partial a_1}\frac{\partial C}{\partial a_2}\frac{\partial C}{\partial a_3}$ and $\sigma'(z^L)$ is just the derivative of the z of the last layer.
This is like finding the 


**2. Error in the Hidden Layers (Recursive Step)**
$$ \delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l) $$

**3. Rate of Change for Biases**
$$ \frac{\partial C}{\partial b^l_j} = \delta^l_j $$

**4. Rate of Change for Weights**
$$ \frac{\partial C}{\partial w^l_{jk}} = a^{l-1}_k \delta^l_j $$

---
**Legend:**
* $L$: The last layer index.
* $l$: The current layer index.
* $\odot$: Hadamard Product (Element-wise multiplication).
* $T$: Transpose (swapping rows and columns).
* $z$: Weighted input (before activation).
* $\sigma'$: Derivative of the activation function (sensitivity).

## Steps for backpropogation

### Phase 1: Forward Pass (The Prediction)

First, we have to see what the network currently thinks.
#### 1. Input: Feed the input vector $x$ into the first layer.

#### 2. Feedforward: For each layer $l = 2, 3, \dots, L$, calculate the weighted input ($z$) and the activation ($a$):
$$z^l = w^l a^{l-1} + b^l$$
$$a^l = \sigma(z^l)$$

(Repeat this until you reach the final output layer $L$).

---
### Phase 2: Backward Pass (The Blame Game)

Now we calculate the errors, starting from the end and moving backward.
- $\delta^l$ is the error at layer l.
- $\delta^{l-1}$ is the error backpropagated to layer l−1.
#### 3. Compute Output Error ($\delta^L$):
Calculate how wrong the final layer was.

$$\delta^L = \nabla_a C \odot \sigma'(z^L)$$
We use these errors to find the derivative of the cost function w.r.t. a weight in the last layer. This will give us one component per weight of the final gradient vector. As we saw in 3b1b's tutorial, finding the $\frac{\partial C}{\partial w^L}$ we get $\frac{\partial C}{\partial a^L}.\frac{\partial a^L}{\partial z^L}.\frac{\partial z^L}{\partial w^L}$ which in the end turns out to be $(a^L-y^L).\sigma'(z^L).a^{L-1}$. 
But this is for one image, to find the actual gradient we add up the values given by other images, to find the most correct direction to descent to.

__Gradient=Destination Error×Source Activation__

#### 4. Backpropagate Error ($\delta^l$):

For each layer from $L-1$ down to $2$, calculate the error using the error from the next layer.
Formula:
$$\delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$$
 The gradient is exactly the same for other layers, __but we do not get the error directly by the cost function.__ Let us see how we get that destination error(this was $(a^L-y^L).\sigma'(z^L)$ for the last layer) and why it works to find our gradient component for the inner layers. ^51e272

Finding the derivative of the cost function w.r.t a weight in the second last layer. Cost doesn't directly depend on this weight but this changes the activation in the neuron in the 2nd last layer which affects the final activation and thus the cost.
So the derivative will be
$$\frac{\partial C}{\partial w^{L-1}}=\frac{\partial C}{\partial a^L}.\frac{\partial a^L}{\partial z^L}.\frac{\partial z^L}{\partial a^{L-1}}.\frac{\partial a^{L-1}}{\partial z^{L-1}}.\frac{\partial z^{L-1}}{\partial w^{L-1}}$$

Solving all the derivatives
$$\frac{\partial C}{\partial w^{l-1}} 
= 
\frac{\partial C}{\partial a^{(l)}} \,
\sigma'\!\left(z^{(l)}\right) \,
w^{l} \,
\sigma'\!\left(z^{(l-1)}\right) \,
a^{(l-2)}.
$$

for $w^{L-2}$ the derivative will be even longer.
We take $\frac{\partial C}{\partial a^L}.\sigma'(z^{(l)})$ as $\delta^L$ error. That is the error given by the last layer
So derivative becomes
$$
\frac {\partial C}{\partial w^{l-1}}=\delta^L.w^l.\sigma'(z^{l-1}).a^{l-2}
$$
According to [[#^51e272|this]] formula we can replace $\delta^L.w^l.\sigma'(z^{l-1})$ with $\delta^l$ 
so finally our derivative becomes
$$
\frac{\partial C}{\partial w^{l-1}}=\delta^{l-1}.a^{l-2}
$$

---

### Phase 3: The Update (The Learning)

Now that we have the "Error Map" ($\delta$) for the whole network, we calculate the gradients and update the parameters.

5. Calculate Gradients:

Use the errors ($\delta$) to find the gradient for every weight and bias.

- **Bias Gradient:** $\frac{\partial C}{\partial b^l} = \delta^l$
    
- **Weight Gradient:** $\frac{\partial C}{\partial w^l} = \delta^l (a^{l-1})^T$
    

6. Gradient Descent (Update Weights):

Update the weights and biases to reduce the error.

$$w^l \leftarrow w^l - \eta \frac{\partial C}{\partial w^l}$$

$$b^l \leftarrow b^l - \eta \frac{\partial C}{\partial b^l}$$
__There are many types of gradient descent. We mainly take small batches of images randomly, and descent by taking the average gradient by them all. This makes sure that the network doesn't go off track because of some outliers. We can also take steps by averaging the gradient over all of the training data but that is expensive, it has the most accurate steps.__

---

### Summary for your code loop:

1. **Forward:** Compute all $z$s and $a$s.
    
2. **Backward:** Compute $\delta^L$, then loop back to find all $\delta^l$s.
    
3. **Gradients:** Use $\delta$s to find $\frac{\partial C}{\partial w}$ and $\frac{\partial C}{\partial b}$.
    
4. **Step:** Subtract (Learning Rate $\times$ Gradient) from current weights.
    

