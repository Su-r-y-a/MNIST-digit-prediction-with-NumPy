# 🧠 MNIST Neural Network from Scratch (NumPy Only)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Su-r-y-a/MNIST-digit-prediction-with-NumPy/blob/main/digit_prediction_nn.ipynb)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/Library-NumPy-013243.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-94.3%25-brightgreen.svg)

An experimental implementation of a fully connected deep neural network built entirely **from scratch using only NumPy** to classify handwritten digits from the MNIST dataset. 

No high-level deep learning frameworks (TensorFlow, PyTorch, or Keras) were used for model building, forward propagation, or backpropagation. TensorFlow/Keras was strictly used as a lightweight utility to load the raw dataset.

---

## 📌 Features

- **Built from First Principles:** Explicit implementation of matrix operations, activation functions, forward passes, cross-entropy loss, and backpropagation gradients.
- **Pure NumPy Vectorization:** Fully vectorized forward and backward passes for efficient matrix operations without explicit loops over samples.
- **High Accuracy:** Achieved **94.3% classification accuracy** on test data within 200 epochs despite a minimal architecture.
- **Visual Evaluation:** Includes sample predictions visualized using `matplotlib`.

---

## 🏗 Network Architecture

| Layer | Type | Input Size | Output Size | Activation |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Flattened Image | $28 \times 28 = 784$ | $784$ | None |
| **Hidden Layer 1** | Dense (Fully Connected) | $784$ | $10$ | **ReLU** |
| **Output Layer** | Dense (Fully Connected) | $10$ | $10$ | **Softmax** |

---

## 🧮 Mathematical Details

### 1. Forward Propagation
* **Hidden Layer:**
  $$Z_1 = W_1 A_0 + b_1$$
  $$A_1 = \text{ReLU}(Z_1) = \max(0, Z_1)$$

* **Output Layer:**
  $$Z_2 = W_2 A_1 + b_2$$
  $$A_2 = \text{Softmax}(Z_2) = \frac{e^{Z_2}}{\sum e^{Z_2}}$$

### 2. Backpropagation & Optimization
Calculates exact partial derivatives using the Chain Rule to update weights ($W$) and biases ($b$) via Gradient Descent:

$$\text{d}Z_2 = A_2 - Y$$
$$\text{d}W_2 = \frac{1}{m} \text{d}Z_2 A_1^T, \quad \text{d}b_2 = \frac{1}{m} \sum \text{d}Z_2$$
$$\text{d}Z_1 = (W_2^T \text{d}Z_2) * \text{ReLU}'(Z_1)$$
$$\text{d}W_1 = \frac{1}{m} \text{d}Z_1 A_0^T, \quad \text{d}b_1 = \frac{1}{m} \sum \text{d}Z_1$$

---

## 📊 Performance & Results

- **Epochs:** 200
- **Final Test Accuracy:** **94.3%**

### Sample Output Predictions
The model generates visual predictions comparing predicted digit labels against actual ground truth images using Matplotlib.

---

## 🚀 How to Run

### Option 1: Open Directly in Google Colab
Click the badge below to run the notebook directly in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Su-r-y-a/MNIST-digit-prediction-with-NumPy/blob/main/digit_prediction_nn.ipynb)

### Option 2: Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/Su-r-y-a/MNIST-digit-prediction-with-NumPy.git](https://github.com/Su-r-y-a/MNIST-digit-prediction-with-NumPy.git)
   cd MNIST-digit-prediction-with-NumPy

## 📦 Installation & Dependencies

To run this project locally, install the required Python libraries using `pip`:

```bash
pip install numpy matplotlib tensorflow
```

> **Note:** `tensorflow` is used solely for importing the MNIST dataset via `tf.keras.datasets.mnist`. All neural network computations (forward pass, loss, backpropagation, and weight updates) are performed strictly using `numpy`.
>
> ## 🛠 Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3** | Core programming language |
| **NumPy** | Matrix multiplication, parameter initialization, activation functions, and gradient descent calculations |
| **Matplotlib** | Visualizing sample MNIST handwritten digits and plotting predictions |
| **TensorFlow / Keras** | Downloading and splitting the raw MNIST dataset |
| **Google Colab** | Cloud-based interactive development environment |
