# TvA calibration

The notebook `main.ipynb` contains code to calibrate ImageNet models, display reliability diagrams and study overfitting. Please start with this.

`calibrators.py` contains code for calibration methods. Implemented scaling methods are temperature scaling, vector scaling, and Dirichlet calibration, with and without TvA and regularization. Binary methods rely on the library netcal, and our code can use these methods with one-versus-all (the standard multiclass to binary reformulation) or top-versus-all (TvA).

`evaluation.py` contains code to compute ECE (equal-size or equal-mass bins), accuracy, average confidence, AUROC, and Brier score.
