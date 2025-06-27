# python code to plot ROC

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import argparse

# check argument if exist
parser = argparse.ArgumentParser()
parser.add_argument('--csv', default='ensemble_result.csv', help='Path to CSV file containing results')
parser.add_argument('--outfile', default='roc', help='Path to output file')
args = parser.parse_args()

# Load the ensemble results
results = pd.read_csv(args.csv)

# Extract truth and predicted columns
truth = results['truth']
# convert labels to numeric
truth = label_binarize(truth, classes=['n', 'p'])

predicted = results['predicted']
predicted = label_binarize(predicted, classes=['n', 'p'])

# Calculate ROC curve
p = results['p']
fpr, tpr, _ = roc_curve(truth, p)
roc_auc = auc(fpr, tpr)

print(f"ROC AUC: {roc_auc}")
# Plot ROC curve
plt.figure()
# display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display = RocCurveDisplay.from_predictions(truth, p, plot_chance_level=True)
display.plot()
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.savefig(f'{args.outfile}')
plt.close()