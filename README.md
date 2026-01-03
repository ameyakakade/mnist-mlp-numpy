# MNIST digit recognizer using Multilayer Perceptron in NumPy
Raw implementation of a multilayer perceptron from scratch that classifies digits in the MNIST dataset.
Digits are handwritten 28×28, 8-bit grayscale images. The network predicts the digit class (0–9) for each image.

The neural network is trained using backpropagation and batch gradient descent, without relying on high-level frameworks that hide the math.
TensorFlow is used only to load the MNIST database. 

[This](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) series by 3b1b on neural networks was inspiration for this project.

### Model Architecture
| Layer  | Details                             |
| ------ | ----------------------------------- |
| Input  | 784 neurons (28×28 flattened image)|
| Hidden | 3 hidden layers, 20 neurons each|
| Output | 10 neurons, output values from 0-1 |

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
Data from MNIST database is loaded as NumPy arrays, flattened and normalised.
Cost is calculated using Mean Squared Error (MSE).


Code is written to maximize readability over performance.
For notes on the calculus (derivatives, chain rule, and cost functions) used in this project, please refer to the docs/ folder.
#### Make sure to use a LaTeX compatible viewer (like obsidian or VS Code).
