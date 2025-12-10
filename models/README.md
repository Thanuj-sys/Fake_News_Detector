# Models Directory

This directory contains trained model files.

## Note
The pre-trained model file (`fake_news_detector.pkl`) is **not included in the repository** because it exceeds GitHub's file size limit (164 MB).

## To Create the Model

Run the training script to generate the model:

```powershell
python train_model.py
```

This will:
- Load the dataset from `data/fake_news_dataset.csv`
- Train all detection components (Markov, TF-IDF, BERT, Web verification)
- Save the trained model as `fake_news_detector.pkl` in this directory
- Training takes approximately 1-5 minutes depending on your system

## Alternative: Use Without Pre-trained Model

The web UI (`web_ui.py`) will automatically train a model on startup if none exists.
