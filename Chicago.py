import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import time
#crime_df=pd.read_csv('https://data.cityofchicago.org/api/views/w98m-zvie/rows.csv?accessType=DOWNLOAD') read csv file and do operations on it (pandas function )
crime_df=pd.read_csv('https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD')

#Checking the first five rows from the dataset
crime_df.head()

#shape() return number of features(22) and records(700000) of the data (700000,22)
crime_df.shape

crime_df.dtypes #Checking datatypes to see if we need any type conversions
crime_df.describe()


# # Preprocessing 
# * Filter for crimes occuring in 2020
# * Concatenate Primary_Type and Description
# * Remove Primary_Type categories that occured < 1000 times
# * Remove missing values

crime_df = crime_df[crime_df['Year'] == 2020] #Dropping the 2020 records as they are not complete yet

crime_df.columns = crime_df.columns.str.replace(' ', '_') #repalcing  the spcaes with _ so we can refer to them with ease

crime_df["Primary_Type_Description"] = crime_df["Primary_Type"] + " " +  crime_df["Description"] #Creating a new column to store description of the crime with the type and description

crime_df['Community_Area'] = crime_df['Community_Area'].astype(object)

crime_df['freq'] = crime_df.groupby('Primary_Type')['Primary_Type'].transform('count') #For plotting we ground by crimes by primrary type
crime_df = crime_df[crime_df.groupby('Primary_Type').freq.transform(len) > 1000] #Getting values with total greater than 1000

crime_df.isna().sum() #Checking for null values

crime_df.isna().sum() /len (crime_df) *100  #Checking for the precentage of null values
# The null value percentage is very nominal we can drop them

crime_df = crime_df.dropna() #Dropping

print('Number of Observations:', crime_df.shape) 
# we dropp nominal values so we have now features(24) because we create new coloumns and records(197514) of the data (197514,24)
unique, counts = np.unique(crime_df.Primary_Type, return_counts = True)
#unique -> return the unique crime checked by 'primary type'  there are 15 different 
#counts indexes of unique numbers (primary type)
print('Number of Unique Crimes:',  len(unique)) #Counting number of type of crimes

unique, counts = np.unique(crime_df.Beat, return_counts = True)
print(len(unique)) #This is our target variable we need to get the bins from the beats data type to get the categories that we want to predict

temp_df=crime_df
temp_df.Beat.astype(object) #First we convert the data to object as we have categorical bins to get preidcted

temp_df['Beat_Bins']=pd.cut(temp_df['Beat'],bins=274,right = True,labels=np.arange(1,275,1))
#Creating new column Beats_bin to store the bins values, we create 274 bins, right==True means that we need to include the right limit while creating the bins , labels would get the 274 labels returned in place of the bins value as we are intersted in categories we must give this argument
#This was our target variable the thing that we want our model to predict
#The values were in numerical order and we wanted it be a label so we can classify it


# # Model Preparation
# From exporing the data, many columns appear to be collinear to one another, and thus, inlcuding all predictors would be redundant and may not provide any extra information, just increase our computational time! For example, IURC (Illinois Unifrom Crime Reporting code) is said to be "Directly linked to the Primary Type and Description".
# 
# Additionally, FBI Code is a variable that describes the type of crime. Because our aim is to predict the type of crime (Primary_Type), it is also important to remove any variables that may leak information about our primary outcome.
# 
# The following steps were performed in order to further process the data for modeling:
# 
# * Select columns of interest - Primary_Type, Arrest, Domestic, Location_Description, and Community_Area.
# * Divide dataset into inputs (crime_x) and output variable (crime_y)
# * One-hot encode categorical inputs in crime_x
# * Recode crime_y to convert data from categorical labels to numeric
# * Shuffle & Split data into training and testing sets using a 90/10% split. 
# 

# Step 1
crime_model = temp_df[['Beat_Bins', 'Location_Description', 'Arrest', 'Domestic', 'Community_Area','Primary_Type']] #We take only meaningful data

# Step 2
crime_x = crime_model[['Location_Description', 'Arrest', 'Domestic', 'Community_Area','Primary_Type']] 
#crime_x independent variables (input)
crime_y = crime_model[['Beat_Bins']]
#crime_y dependent variable (we want to predict this)

# First we see the top 10 crimes, then we visulize the hotspot for crime

crime_df10 = crime_model.Primary_Type.value_counts()
crime_df10 = crime_df10.head(10)
crime_df10 = pd.DataFrame(crime_df10)

plt1 = crime_df10.plot(kind="bar", color = "tomato")
plt1.tick_params(axis="x", labelsize = 10, labelrotation = 90)
plt1.set_title("Top 10 Crimes")
plt1.get_legend().remove()
#################################################################
crime_df10 = crime_model.Location_Description.value_counts()
crime_df10 = crime_df10.head(10)
crime_df10 = pd.DataFrame(crime_df10)

plt1 = crime_df10.plot(kind="bar", color = "tomato")
plt1.tick_params(axis="x", labelsize = 10, labelrotation = 90)
plt1.set_title("Top 10 Locations")
plt1.get_legend().remove()
##########################################################3
crime_df10 = crime_model.Community_Area.value_counts()
crime_df10 = crime_df10.head(10)
crime_df10 = pd.DataFrame(crime_df10)

plt1 = crime_df10.plot(kind="bar", color = "tomato")
plt1.tick_params(axis="x", labelsize = 10, labelrotation = 90)
plt1.set_title("Top 10 Communities")

crime_y=pd.get_dummies(crime_y).values #get_Dummies would one hot encode the variable
crime_x=pd.get_dummies(crime_x).values

from sklearn.model_selection import train_test_split #Split to train_test
X_train, X_test, y_train, y_test = train_test_split(crime_x, crime_y, test_size=0.10, random_state=42)
# we want out dataset to be 10% test data and 90% training data

#Convert data to numpy array so that the it works with Keras Deep Learning model
X_train=np.asarray(X_train).astype(np.float32)
X_test=np.asarray(X_test).astype(np.float32)
y_train=np.asarray(y_train).astype(np.float32)
y_test=np.asarray(y_test).astype(np.float32)


# # First we create feed forward network for predictions

#Import libraries
from keras.models import Sequential #To get a Neural Network in sequence
from keras.layers import Dense #Dense means that every layers would be linked to each other
import tensorflow as tf
import tensorflow

adam_optim=tf.keras.optimizers.Adam(learning_rate=0.01) #We use adam optimizer with fairly good learning rate so we can coneverge to the target point
model=Sequential(
    [Dense(32, activation='relu',input_dim=X_train.shape[1]) #First layers has 32 neurons
    ,Dense(16, activation='relu'), #Second has 16 with RELU activation function
    Dense(274,activation='softmax')]) #Final layer is equal to the number of possible outcomes with softamx activation

model.compile(loss='categorical_crossentropy',optimizer=adam_optim, metrics=['accuracy']) #We compile the model and use accuracy to get the best model
history=model.fit(X_train,y_train,epochs=100,verbose=1,batch_size=128) #We train the model for 100 epochs and by feeding the data in the batches of 128

# serialize model to YAML
#We save our model so that we can use it at a later stare
model_yaml = model.to_yaml()
with open("modelfeedforward.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("modelfeedforward.h5")
print("Saved model to disk")

#Now we evaluate the model on test set
_,accuracy=model.evaluate(X_test,y_test)
y_pred=model.predict(X_test) #The accuracy is good which is 55.65 percent

import matplotlib.pyplot as plt #See that the loss decrease over time
plt.plot(history.history.get('loss')) 

plt.plot(history.history.get('accuracy'))  #The accuracy increases our time at places it goes fown byt at the last it is highest

model.summary() #Get the textual representation of the structure of our Neural Netwok
tf.keras.utils.plot_model(model,show_shapes=True,expand_nested=True) #We plot the model to see how it look

# # CNN
from keras.layers import Flatten #Import required libraries
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.convolutional import MaxPooling1D
#shape[1]=colomn
x_train = X_train.reshape(-1, 1, X_train.shape[1]) #As CNN is used for image classification and it expects a 3d input and we have divided our data into grid of 274 so we need to change the shape from 2d to 3d so that we can use CNN for classification of non textual data

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu',padding='same',input_shape=(x_train.shape[1],x_train.shape[2])))#Convolutional Input layers with 64 filters and padding=same "same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.
model.add(Conv1D(filters=64, kernel_size=3, activation='relu',padding='same')) #Second 1d convolutional layer
model.add(Dropout(0.5)) #A drop out layer to precent over fitting
model.add(MaxPooling1D(pool_size=2,padding='same')) #A max pooling layer with padding same
model.add(Flatten()) #IN the CNN architecture (image can be found  on google) has Flatten layer to flat the output
#The CNN ends here, the next part is any standard NN, the output from CNN is feeded into simple Network to get the output
model.add(Dense(100, activation='relu')) #2nd last layer with 100 neurons
model.add(Dense(274, activation='softmax')) #Fina layer with 274 as we have 274 output possibilties
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network"
model.fit(x_train, y_train, epochs=100, batch_size=128,verbose=1) #Again we give inout in 128 batches (can be an number) for 100 epochs

model_yaml = model.to_yaml()
with open("modelcnn.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("modelcnn.h5")
print("Saved model to disk")

model.summary() #Observe the parameters increases with the complexity of the model

_,accuracy=model.evaluate(X_test.reshape(-1,1,215),y_test)
y_pred=model.predict(X_test.reshape(-1,1,215)) #The accuracy is 56.17%

plt.plot(history.history.get('loss')) 
plt.plot(history.history.get('accuracy')) 

tf.keras.utils.plot_model(model,show_shapes=True,expand_nested=True) #You can see the architecture visual presnetation here


# # RNN
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense #Again we import the librararies
x_train = X_train.reshape(-1, 1, X_train.shape[1]) #get the 3d shape for input as RNN needs ir
model = Sequential()
model.add(LSTM(100,input_shape=(x_train.shape[1],x_train.shape[2]))) #Fiest layer with 100 neurons and input shape as required
model.add(Dropout(0.5)) #Add drop out to prevent overfitting
model.add(Dense(100, activation='relu'))
#Just like standard RNN with LSTM
model.add(Dense(274, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam'
              , metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=128,verbose=1)

model_yaml = model.to_yaml()
with open("modelrnn.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("modelrnn.h5")
print("Saved model to disk")

model.summary() #Observe the parameters increases with the complexity of thr model

_,accuracy=model.evaluate(X_test.reshape(-1,1,215),y_test)
y_pred=model.predict(X_test.reshape(-1,1,215)) #The accuracy is 56.76%

plt.plot(history.history.get('loss')) 

plt.plot(history.history.get('accuracy')) 

tf.keras.utils.plot_model(model,show_shapes=True,expand_nested=True) #You can see the architecture visual presnetation here


# # Recurrent CNN (RNN+CNN= RCNN)
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D,Conv2D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import  BatchNormalization
from keras.layers import PReLU
#again we import libraries and create the RCNN
model= Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu',padding='same',input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Conv1D(filters=64, kernel_size=3, activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Conv1D(filters=64, kernel_size=3, activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(PReLU())
model.add(Conv1D(filters=64, kernel_size=3, activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(PReLU())
model.add(MaxPooling1D(pool_size=2,padding='same'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(274, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=128,verbose=1,epochs=100)

model_yaml = model.to_yaml()
with open("modelrcnn.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("modelrcnn.h5")
print("Saved model to disk")

model.summary()

_,accuracy=model.evaluate(X_test.reshape(-1,1,215),y_test)
y_pred=model.predict(X_test.reshape(-1,1,215))

plt.plot(history.history.get('loss')) 

plt.plot(history.history.get('accuracy'))
 
tf.keras.utils.plot_model(model,show_shapes=True,expand_nested=True) #You can see the architecture visual presnetation here