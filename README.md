# Titanic Prediction Model

This is just a simple script to train a model on the Titanic dataset. I used a Random Forest classifier because it usually works pretty well for these kinds of things.

## How to run it

You need to have python installed.

1. Install the requirements (basically just pandas and sklearn):
   ```
   pip install pandas scikit-learn
   ```

2. Run the script:
   ```
   python train_model.py
   ```

The script will load the data, handle some missing values, train the model, and then print out the accuracy.

## Files

- `titanic.csv`: The data.
- `train_model.py`: The code to train the model.

That's it!
