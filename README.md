# Dog-Breed-Classifier
Udacity assignment - project 2
This project is about detecting human face and dog presence in any given image.
If there is a dog presence, then the algo will predict the likely canine's breed.
But the funny part of this exercise if, the end output will identify the resembling dog breed :)
Before we come to that, this exercise creates 3 broad parts:

Detect Humans using cv2
Assess the Human Face Detector The submission returns the percentage of the first 100 images in the dog and human face datasets that include a detected, human face.
This provides more than 95% accuracy.

Detect Dogs using VGGNet
Use a pre-trained VGG16 Net to find the predicted class for a given image: dog_detector function returns True if a dog is detected in an image and False if not.
This should provides more than 98% accuracy.

Detect Dog Breed using CNN from scratch / using Transfer Learning from Res50Net
(i) From Scratch - CNN trained model attains at least 10% accuracy on the test set.
(ii) Using Transfer Learning - Model architecture that uses part of a pre-trained model with accuracy on the test set of 60% or greater.
