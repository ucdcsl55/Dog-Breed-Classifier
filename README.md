# Project 2 - Dog-Breed-Classifier

This Repository stores my solution to the "Dog Breed Classifier" problem presented as the second project in the Deep Learning Nano Degree by Udacity.

It is about using cv2 algo to detect human face. It has a good accuracy rate. Then using Vgg16 to develop an algo to detect dog presence in any given image.
Once the Vgg16 algo is developed, we continue by using CNN architecture to predict the likely canine's breed in any given image.

But the funny part of this exercise is.... at the end output of the project, we are asked to identify a human face to a resembling dog breed :)
Before we come to the end, this exercise actually has 3 broad parts:

*****************************************************************************************************

# Part 1 | Detect Humans using OpenCV
## Human Face Detector
The submission returns the percentage of the first 100 images in the dog and human face datasets that include a detected, human face.
### Accuracy Rate
There are 98.0% human faces detected in human_files.
There are 17.0% human faces detected in dog_files.

*****************************************************************************************************

# Part 2 | Detect Dogs using VGG16Net
### Dog Presence Detector
Used a pre-trained VGG16Net model to find the predicted class for a given image.
### Accuracy Rate
There are 1.0% dog images detected in human_files.
There are 100.0% dog images detected in dog_files.


*****************************************************************************************************

# Part 3 | Create a CNN to Classify Dog Breeds
Two approaches applied for better understanding of the algo.

(i) __Scratch__ - apply CNN trained model which has a series of functions. But this scratch model attains only 11% accuracy on the test set.

From formula to coding....

![formula](https://github.com/ucdcsl55/Dog-Breed-Classifier/blob/main/MLP_formula.png?raw=true)
![coding](https://github.com/ucdcsl55/Dog-Breed-Classifier/blob/main/MLP_function.png?raw=true)

  #1. Firstly, examine the shape of the image, which has 3 RGB colors and its a 224X224 pixel image.
  
       - torch.Size([32, 3, 224, 224]). Note 32 is batch size.
       
  #2. Then start by applying convolutional layers, max pooling, dropout, batch normalisation, fully-connected layers and lastly defining the feedforward behaviour.
       
       - Conv2D:: self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding).
           - We can use the spatial dimension formula to keep changing XY through the layers: [(W−K+2P)/S]+1
       - MaxPool2D:: nn.MaxPool2d(kernel_size, stride, padding, dilation, ceil_mode)
           - This is to down-sample any XY size by the stipulated values.
       - Dropout:: nn.Dropout(0.%)  where % denotes the probability dropout to prevent overfitting.
       - Batch Normalisation:: nn.BatchNorm1d(num_features)
           - Batch Normalization allows us to use much higher learning rates and be less careful about initialization. 
           - It also acts as a regularizer, in some cases eliminating the need for Dropout.
       - Fully-Connected Layers: nn.Linear(in-channel, out-channel)
       - Forward:: self.pool(F.relu(self.conv1(x))).
      __Note__ : For simplicity, am using the following standard rules.
                - kernel size of (3,3) and padding 1
                - nn.MaxPool2d(2, 2)
                - nn.Dropout(0.25)  

The detailed logical steps on #2 above are:-
    
    - Applied 3 Convolutional Layers
        - Converts image tensor (224,224,3) -- self.conv1 = nn.Conv2d(3,32,kernel,stride,padding) -- nn.MaxPool2d(2, 2)
            ==> (112,112,16)
        - Converts second level via -- self.conv1 = nn.Conv2d(32,64,kernel,stride,padding) -- nn.MaxPool2d(2, 2)
            ==> (56,56,32)
        - Converst third level via -- self.conv2 = nn.Conv2d(64,128,kernel,stride,padding) -- nn.MaxPool2d(2, 2)
            ==> (28,28,64)

    - Applied 3 Fully-connected Layers
        - First layer will responsible for taking as input of my final downside stack of feature maps above.
        - nn.Linear(224*224, 1024) 
        - Second layer takes in the output from the first layer
        - nn.Linear(1024, 512) 
        - Third layer takes in the output from second layer and produce the final output of 133 breed types.
        - nn.Linear(512, 133) 
        Note, 133 represents the total number of dog breed in the train dataset.

    - Used standard Feedfoward behaviour
        - This takes in a sequence of function actions.
            - First it takes the output from each Convolutional layer, then apply ReLu activation function.
            - Then apply max pooling layer(s), flatten the image and add dropout to prevent overfitting.
            - Last use the batch normalization function.
              - self.pool(F.relu(self.conv1(x))) -- repeat this for each convolutional layer.
              - x.view(-1, x.size(0)) then take the end resultant from the convolutional layers and flattens into a vector shape.
              - self.dropout(x) - adding a dropout layers to prevent overfitting
              - self.batch_norm1 - mainly to make model faster and more stable by re-centering and re-scaling

The final CNN model should look like the expression below:
  ![model_scratch](https://github.com/ucdcsl55/Dog-Breed-Classifier/blob/main/model_scratch.png?raw=true)

*****************************************************************************************************

(ii) __Transfer Learning__ - apply on an existing pre-trained model architecture. This provides an accuracy on the test set of 75%.
  
  The following image provides a good summary of what needs to be modified if the datasets we training are somewhat different from the original data.
      ![Transfer_Learning](https://github.com/ucdcsl55/Dog-Breed-Classifier/blob/main/Transfer_Learning.png?raw=true)

  As this approach is using an existing pre-trained model, it makes the codes lot easier. All we need is to add a last linear layer. 
   
        # Check the number of dog breed in the training dataset
        num_dog_breed = len(train_data.class_to_idx)
        # Update the last layer
        num_features = model_transfer.fc.in_features
        model_transfer.fc = nn.Linear(num_features, num_dog_breed)

*****************************************************************************************************

# Conclusion
This project led me to develop an CNN model architecture from scratch. Details on layers of Conv2D, max pooling, dropout to prevent overfitting, batch normalisation to optimise the steps, apply criterion (MSE, CrossEntropyLoss etc) and optimizer (SGD, Adam etc) and the benefits gained using transfer learning.
