# Ensemble module 

List of ensemble arguments:   

  * *--config*: which experiments (INI files) to combine
  * *--method* (optional): majority_voting, mean (default), max, sum, uncertainty, uncertainty_weighted, confidence_weighted, performance_weighted  
  * *--threshold*: uncertainty threshold (1.0 means no threshold)
  * *--weights*: weights for performance_weighted method (could be from previous UAR, ACC)
  * *--outfile* (optional): name of CSV file for output (default: ensemble_result.csv)
  * *--no_labels* (optional): indicate that no ground truth is given 