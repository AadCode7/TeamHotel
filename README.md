# HMS - Harmful Brain Activity Classification
## Classify seizures and other patterns of harmful brain activity in critically ill patients

Kaggle Competition - https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/overview

* What are Spectrograms?
Spectrograms are images of sound saved as a matrix in *.parquet* files. It shows different frequencies in audio signal change over time. 

Kaggle Spectrograms vs. EEG Spectrograms
In this competition, Kaggle provides us with both spectrograms and raw eeg waveforms. The Kaggle spectrograms are 10 minutes long and the eeg waveforms are 50 seconds. The middle 50 seconds of both data are the same time window observing the same event. (i.e. the middle 50 seconds is same information displayed two different ways). Spectrograms are just visual representations of raw waveforms.

# Libraries  
1. os
2. keras - High-level deep learning API; useful for model creation, layers and training.
3. keras_cv - Keras extension for computer vision; useful for pretrained models, augmentations & loss functions.
4. MixUp - Combines two images and labels
5. RandomCutout - Augmentation to randomly black out square patches of an image to improve robustness.
6. ImageClassifier - Prebuilt model head for classification tasks.
7. torch - Library for Tensor operations & deep learning.
8. one_hot - Convert labels to one-hot encodedd format.
9. pad - Padding to tensors.
10. DataLoader - Efficient way to batch, shuffle & load dataets during training.
11. tqdm - Progress bar during training & evaluation.
12. numpy - Numerical computations & array handling.
13. pandas - Loading and manipulating structred data.
14. pyarrow - Supports Apache Arrow columnar data format; ueful for fast data serialization & reading .parquet files.
15. sklearn - Tools for evaluation, metrics and data splitting
16. StratifiedGroupFold - Ensures that each fold of cross-validation contains the same class distribution(Stratified)
    
