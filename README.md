# Numerical Methods in Data Science Course Projects

Welcome to the repository for the **Numerical Methods in Data Science** course projects. This collection showcases various numerical methods and algorithms applied to data science problems. Each project demonstrates practical implementations and applications of these techniques.

## Table of Contents

- [Project Overview](#project-overview)
- [List of Projects](#list-of-projects)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This repository contains Jupyter notebooks and python scripts that explore fundamental numerical methods used in data science. The projects cover a range of topics from linear algebra techniques to optimization algorithms, providing hands-on experience with key concepts.

## List of Projects

1. [**Singular Values Decomposition (SVD)**](./notebooks/svd_nb.ipynb)
   An exploration of SVD, its properties, and applications in data reduction.

2. [**Moore-Penrose Pseudo Inverse**](./notebooks/mppseudoinv_nb.ipynb)
   Implementation of the Moore-Penrose pseudo inverse for solving linear systems.

3. [**Solving a System of Equations using Moore-Penrose Pseudo Inverse**](./notebooks/system_of_equation_solver_mppi_nb.ipynb)
   Demonstrates how to solve linear equations using the pseudo inverse method.

4. [**Classic Gram-Schmidt Process**](./notebooks/classic_gram_schmidt_nb.ipynb)
   Implementation of the Classic Gram-Schmidt algorithm for orthogonalization.

5. [**Modified Gram-Schmidt Process**](./notebooks/modified_gram_schmidt_nb.ipynb)
   An improved version of the Gram-Schmidt process for numerical stability.

6. [**Gradient Descent Algorithm**](./notebooks/gradient_decent_nb.ipynb)
   A practical implementation of the gradient descent optimization algorithm.

7. [**Image Compression using SVD**](./notebooks/image_svd_nb.ipynb)
   Image compression technique utilizing SVD for matrix approximation, employing the Elbow Method to determine the optimal number of singular values.

8. [**Linear Regression using Moore-Penrose Pseudo Inverse with K-Fold Cross-Validation**](./blob/main/notebooks/linear_regression_nb.ipynb)
   A linear regression model built using the pseudo inverse, evaluated with K-Fold cross-validation.

9. [**Google's PageRank Algorithm**](./notebooks/page_rank_nb.ipynb)
   Implementation of the PageRank algorithm, a fundamental algorithm used by Google for ranking web pages.

## Getting Started

To get started with this repository, clone it to your local machine:

```bash
git clone https://github.com/yourusername/numerical-methods-data-science.git
cd numerical-methods-data-science
```

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Required libraries (NumPy, SciPy, OpenCV, etc.)

You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## Usage

Each project is contained within its own Jupyter notebook. Open the notebooks in Jupyter to explore the implementations and run the code interactively.

```bash
jupyter notebook
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or additional projects, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
