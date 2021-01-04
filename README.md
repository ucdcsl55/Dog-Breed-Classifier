# Project 2 - Dog-Breed-Classifier from Udacity 

This project is about detecting human face and dog presence in any given image.
If there is a dog presence, then the algo will predict the likely canine's breed.
But the funny part of this exercise is.... at the end output of the project, we are asked to identify a human face to a resembling dog breed :)
Before we come to the end, this exercise actually has 3 broad parts:

#Detect Humans using cv2

Assess Human Face Detector.
The submission returns the percentage of the first 100 images in the dog and human face datasets that include a detected, human face.
This provides more than 95% accuracy.

#Detect Dogs using VGGNet

Use a pre-trained VGG16Net model to find the predicted class for a given image.
This should provides more than 98% accuracy.

#Detect Dog Breed using CNN from scratch / using Transfer Learning from Res50Net
Two approaches applied for better understanding of the algo.
(i) From Scratch - apply CNN trained model which has a series of functions. But this scratch model attains only 11% accuracy on the test set.
(ii) Using Transfer Learning - apply on an existing pre-trained model architecture. This provides an accuracy on the test set of 72%.
