# Machine-Learning---Forcasting-Crime-project
#Forecasting crime with deep learning

##Please replace yammal with Json, because Google's TensorFlow drops YAML support due to code execution flaw, this is an URL for article that discusses it: 
https://www.bleepingcomputer.com/news/security/googles-tensorflow-drops-yaml-support-due-to-code-execution-flaw/

Introduction:
The project focuses on training of the different Deep Learning neural network to predict the Police Beats of a crime that is occurred in the Chicago city. The Chicago Crime data from the year 2001 to 2019 is used for this purpose.
Data preprocessing:
The first step in the creation of any deep learning architecture is the preprocessing step in which the data is cleaned, to feed in the correct and most accurate data by feature selection into the algorithm. As the models can only work on numerical data, we converted the categorical data to their numerical alternative. The NAN values in the data are dropped because their proportion was smaller as compared to the whole dataset. 
Feature Selection
Using correlation analysis, we handpicked few of the features that we would be using to build the model, these features have either close positive or close negative correlation with the target variable that is Beat
•	Location_Description
•	Arrest
•	Domestic
•	Community_Area
Data visualization 
As our focus was on the creating deep learning model so we limited our data visualization but the visualizations that we used gives great insights into the data
![image](https://user-images.githubusercontent.com/98653093/158594111-ac3d3fab-7aa7-48be-8ad5-193eb994ecf7.png)

![image](https://user-images.githubusercontent.com/98653093/158594139-2f120e05-8904-4d05-a82c-9a37dc28f485.png)



Train and test split
The next step is splitting our data into train and test split, the train set is used to train the model and it has 90% of the data whereas the rest of 10% is used for evaluation of the models.
Model used
•	Feed Forward Network
•	Convolutional Neural Network
•	Recurrent Neural Network
•	Recurrent Convolutional Neural Network


Feed Forward Network
•	Every neuron is connected with each other by the dense layer.
•	The input is given in batches
•	We use SoftMax activation function at the output layer because the output can belong to 274 different classes
•	Adam Optimizer is used to reduce the loss
•	It has one input layer with two hidden and one output layer all connected in sequential manner every neuron passes information to the other and in the last the output is computed.


Convolutional Neural Network
•	Every neuron is connected, and the input is given in batches.
•	The first layer is the Convolution layer that applies the same padding to the input with a filter of 64 and performs the convolution operation, the output of that layer  that is computed via SoftMax is  connected with the first hidden layer that again narrow the input down to another filter of 64 and applies the same padding with the SoftMax function, then a dropout layer is used a regularization parameter to prevent overfitting in the next layer the input is applied the Max Pool function that creates the feature map of the input.
•	The output of Max Pool is flattened to 2-D so that it can be used as an input to the feed forward network that predicts the class.


Recurrent Neural Network
•	The first layer has a short memory by which it remembers the sequence the input in again given to the same layer so that it can learn better. A dropout regularization is applied to prevent overfitting the output of the layer is then given into a Feed Forward Network that predicts the class


Recurrent Convolutional Neural Network
•	This a combination of CNN and RNN, the algorithm combines both features of RNN and CNN, Convolutional layer is applied to get more focused on the target point and then the power of RNN is used to feed to value back into the network to learn the parameters better at the end a feed forward network in connected to predict the class.

For the last 3 architectures we transformed the data from 2-D to 3-D as they expect the data to be in an input shape of 3D.
Conclusion
It has been found out that RCNN performed better than any other algorithm in prediction that police beat in which the crime would occur depending upon the nature and characteristics of the crime

This is an images for compiling the code that shows how the prediction get more percise each time we use anoher algorithm: 

![image](https://user-images.githubusercontent.com/98653093/158594711-17dd9bd0-4545-44d7-a243-b9394d757ca3.png)


![image](https://user-images.githubusercontent.com/98653093/158594857-41146415-c2bb-4c54-80d2-530c42256c82.png)

