# Modifying LIME for Neural Networks on Medical Imaging

This project aims at explaining the decisions of the 141-layer CheXNet model developed by [Rajpurkar (2017)](https://stanfordmlgroup.github.io/projects/chexnet/). We modify the definition of neighbours in LIME for interpreting a deep neural network on chest X-Rays. 

This project was completed as part of Machine Learning for Healthcare class at MIT ([6.871/HST.956](https://mlhcmit.github.io)). Full results are available in `final_report.pdf`. Colaborators: Jiong Wei Lua, Alexandru Socolov and Andras Szep. 

## Approach
We improve upon [classical LIME](https://github.com/marcotcr/lime) by defining clinically meaningful neighbors. We explore two definitions of neighbors:

1. Similarity in Variational Auto Encoder latent space (done on ChestX-ray14 dataset). 
2. Similarity in medical reports attached to each patient using BERT emeddings. 

## Model to be explained
[CheXNet](https://stanfordmlgroup.github.io/projects/chexnet/) takes a chest X-Ray and outputs a probability for each of 14 deceases. The model has been shown to achieve an AUC of 0.7-0.9 on different diagnoses. It has been trained on the ChestX-ray14 dataset. 

## Data
Two sources are used: 

1. [MIMIC CXR from PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/) contains 377,110 images corresponding to 227,835 radiographic studies. 14 labels were extracted using two free-text methods. **Advantage**: has medical records attached to each image. 
2. [ChestX-ray14 from NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC) has 112,120 frontal-view chest X-ray images of 30,805 unique patients. 14 labels extracted using two free-text methods. **Advantage**: has a pretrained CheXNet model freely available
