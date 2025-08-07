# ADHD Dataset Decision Tree Classifier with SFFS

This repository contains four Python scripts that apply Decision Tree Classification using SFFS (Sequential Forward Floating Selection) and Leave-One-Out Cross Validation (LOOCV) to ADHD-related datasets.

## Project Files
- **zigzag_trace_dt.py**: Decision Tree Classification with SFFS - ZigZag Trace Dataset.
- **zigzag_predict_dt.py**: Decision Tree Classification with SFFS - ZigZag Predict Dataset.
- **pl_trace_dt.py**: Decision Tree Classification with SFFS - PL Trace Dataset.
- **pl_predict_dt.py**: Decision Tree Classification with SFFS - PL Predict Dataset.

## Requirements

- Python 3.x
- scikit-learn
- pandas
- mlxtend
- graphviz
- Google Colab (for Google Drive integration)

You can install the dependencies using:

```bash
pip install scikit-learn pandas mlxtend graphviz
```

## How to Use

1. Mount Google Drive in Google Colab.
2. Upload your CSV files to the appropriate location.
3. Run each script to perform feature selection and train a Decision Tree.
4. Results will be saved to your Drive in the `result` folder.

## Notes

- Each script performs grid search to optimize Decision Tree parameters.
- Uses SFFS with LOOCV for feature selection.
- Make sure the paths to the CSV files and output directories exist in your Drive.
- This project does not require Google Drive or Google Colab integration. All file paths are local, and the code can be run in any Python environment.
