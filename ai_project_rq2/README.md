# Artificial Intelligence Project - Research Question 2

**Author**: [TAKOUDJOU NDE KENNETH]
**Course**: Artificial Intelligence Project

---

## 1. Project Overview

This project addresses Research Question 2 of the Artificial Intelligence Project, which is:

> *If we replace SVD++ with a neural collaborative model (DeepFM) — used alone and as the CF backbone in the same tag-based hybrid — do we outperform ItemKNN and Popularity (Top-K Recall@K, NDCG@K) on implicit Steam data, and how do Catalog Coverage, Diversity, and Popularity Bias change?*

The project is structured as a series of Python scripts to ensure a professional, modular, and reproducible workflow, as required by the lecturer. The code was originally developed in a Jupyter Notebook and has been refactored into this file-based structure.

## 2. File and Folder Structure

The project is organized into the following logical structure:

/ai-project-rq2/
├── .gitignore               # Specifies files to be ignored by Git (e.g., data, cache files).
├── README.md                # Project documentation (this file).
├── requirements.txt         # Lists all required Python packages to run the project.
├── src/                     # Source code directory.
│   ├── init.py          # Marks src as a Python package.
│   ├── config.py            # Defines all project constants and configuration settings.
│   ├── data_prep.py         # Handles data loading, cleaning, and preprocessing.
│   ├── models.py            # Contains the logic for training the recommendation models.
│   ├── evaluation.py        # Implements all scoring functions and metric calculations.
│   └── main.py              # The main script that runs the entire pipeline.
└── data/                    # Directory for datasets.
├── games.csv            # Metadata about games (tags, genres).
├── train_interactions.csv # Training interaction data (user-item plays).
├── test_interactions_in.csv # Test data for final top-K recommendations.
└── ... (output files are saved here)

## 3. Requirements

To run this project, you need a Python environment with the following dependencies. It is highly recommended to use a virtual environment like `conda` (as used in the `kenneth` environment) or `venv` to manage these dependencies.

1.  **Python 3.7+** (Libreco compatibility).
2.  **Required Libraries**: The following libraries can be installed using `pip`.
    ```bash
    pip install libreco==1.* scikit-learn pandas numpy matplotlib tqdm tensorflow==1.*
    ```
    *Note: Libreco v1.* requires TensorFlow v1.* or TensorFlow 2.* with `tf.compat.v1` support.*

## 4. How to Run the Project

1.  **Clone the Repository**: Clone this project from GitHub to your local machine.
    ```bash
    git clone [https://github.com/your-username/ai-project-rq2.git](https://github.com/your-username/ai-project-rq2.git)
    cd ai-project-rq2
    ```
2.  **Install Dependencies**: Install all required packages using `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare Data**: Place your `games.csv`, `train_interactions.csv`, and `test_interactions_in.csv` files into the `data/` directory.
    * **Crucial Step**: You must update the `DATA_DIR` path in `src/config.py` to point to the absolute path of your `data/` directory. For example: `DATA_DIR = r"C:\Users\YourName\Documents\ai-project-rq2\data"`.
4.  **Run the Main Script**: Execute the `main.py` script from your terminal.
    ```bash
    python src/main.py
    ```

## 5. What Each File Does

* **`config.py`**: Here, I am defining all the constants. This includes file paths, hyperparameters for DeepFM, and settings for the hybrid model. Changing a value here affects the entire project run.
* **`data_prep.py`**: Here, I am handling the data. This script loads the raw CSV files, performs data cleaning and normalization, splits the data into train/validation/test sets, and builds the necessary Libreco `Dataset` objects. It also prepares the TF-IDF content features.
* **`models.py`**: Here, I am training the models. This file encapsulates the training logic for both the ItemCF and DeepFM models. It includes the hyperparameter tuning loop for DeepFM, which finds the best configuration before training the final model.
* **`evaluation.py`**: Here, I am evaluating the models. This file contains all the functions to calculate the core metrics: Recall@K, NDCG@K, Hit Rate, Catalog Coverage, Diversity, and Popularity Bias. It also implements the blending logic for the hybrid model.
* **`main.py`**: Here, I am running the show. This script orchestrates the entire pipeline. It calls the functions from the other modules in the correct sequence to perform data preparation, model training, evaluation, plotting, and result saving.

## 6. How to Organize Work on GitHub

* **Branches**: For a single-person project, the `main` branch is sufficient. You don't need to create separate branches unless you are working on a major new feature or bug fix and want to keep it separate from your main work.
* **Commits**: Make small, frequent, and descriptive commits. For example, a commit message like "feat: Add DeepFM hyperparameter tuning logic" is much more helpful than "changed some code."
* **Issues**: You can use GitHub's Issues feature to track your progress, log bugs, or note down future enhancements.

## 7. Sanity Tests

This project includes built-in sanity checks and print statements to provide concrete feedback at each step.

* **`data_prep.py`**: Prints the number of rows, unique users, and unique items after cleaning, which is a good sanity check for data integrity.
* **`models.py`**: The hyperparameter tuning loop provides feedback on each trial's performance, confirming that the models are learning and that the tuning process is functional.
* **`evaluation.py`**: All evaluation functions are encapsulated, ensuring consistency. The final summary table and plots serve as a comprehensive sanity check on the relative performance of the different models.

The entire script is designed to run sequentially, and the final output files and plots serve as the ultimate proof that the code runs correctly and produces the required results for the research question.

---