# Hybrid Recommender System for Games

This project implements a hybrid recommendation system that combines collaborative filtering (SVD++) and content-based filtering to provide personalized recommendations for games.

## Project Structure

The code is organized into a modular and professional structure for clarity and maintainability.

- `data/`: Contains the raw input data files.
- `results/`: Stores all output from the scripts, including plots and evaluation summaries.
- `config.py`: A central file for all global settings, file paths, and hyperparameters.
- `data_prep.py`: Handles all data loading, cleaning, and preparation tasks.
- `models.py`: Contains the logic for defining and training the recommendation models (SVD++ and ItemCF).
- `evaluation.py`: Implements all the evaluation metrics, plotting functions, and the logic to assess model performance.
- `main.py`: The main entry point for the entire pipeline. It orchestrates the flow from data preparation to final recommendations.
- `requirements.txt`: Lists all the necessary Python libraries for this project.
- `.gitignore`: Specifies which files and directories should be ignored by Git.

## Requirements to Run the Code

### 1. Python Environment

You need to have Python installed. It is highly recommended to use a virtual environment to manage dependencies. This project was developed and tested using a Conda environment.

### 2. Install Dependencies

First, navigate to the project's root directory. Then, install the required libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt