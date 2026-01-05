# MNIST digit recognizer using Multilayer Perceptron in NumPy
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-required-orange)
![Python](https://img.shields.io/badge/Python-3.9+-blue)

A from-scratch NumPy implementation of a multilayer perceptron trained on MNIST, built to understand backpropagation at the mathematical level.

__Accuracy: ~93% on the MNIST test set__

### Why This Exists
The neural network is trained using backpropagation and batch gradient descent, without relying on high-level frameworks that hide the math.
TensorFlow is used only to load the MNIST dataset. Code is written to maximize readability over performance.

[This](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) series by 3b1b on neural networks was inspiration for this project.

### Model Architecture
| Layer  | Details                             |
| ------ | ----------------------------------- |
| Input  | 784 neurons (28×28 flattened image)|
| Hidden | 3 hidden layers, 20 neurons each|
| Output | 10 neurons, output values in range [0,1] |

<details>
<summary><h2>Screenshots</h2></summary>
<p>Sample digit prediction and confusion matrix after training.</p>


<img width="376" height="330" alt="Screenshot 2026-01-03 at 9 33 48 PM" src="https://github.com/user-attachments/assets/5dfe4f7e-1fe8-4420-ae3a-5a06d8e9210f" />
<img width="393" height="421" alt="Screenshot 2026-01-03 at 9 34 54 PM" src="https://github.com/user-attachments/assets/cccd9309-dfe7-4535-aa63-26fb5f48bcae" />

![IMAGE 2026-01-05 22:17:41](https://github.com/user-attachments/assets/d8967e4a-5e33-40f1-8c8c-2ffee5bcf942)
</details>

### Usage
Make a virtual environment and install requirements given in requirements.txt
```
python3 -m venv .venv
pip install -r requirements.txt
```
Run `train.py`. It will save the data (weights and biases) to `brain.npz`
(database is downloaded automatically using TensorFlow)
After training, run `usage.py`. This can be used to find accuracy, confusion matrix and sample predictions.

### The math
Weights and biases are randomly initialised.
The sigmoid function is used as the activation function for all layers.
Data from MNIST database is loaded as NumPy arrays, flattened and normalized.
Cost is calculated using Mean Squared Error (MSE).

### Limitations
- Uses sigmoid instead of ReLU.
- Uses MSE instead of cross-entropy
- Not optimized for performance

For notes on the calculus (derivatives, chain rule, and cost functions) used in this project, please refer to the docs/ folder.
#### Make sure to use a LaTeX compatible viewer (like obsidian or VS Code).
