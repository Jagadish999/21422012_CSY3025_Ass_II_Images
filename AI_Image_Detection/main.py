import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

#Path of folder containing training and testing images
path_for_images_training = './data/train'
path_for_images_testing = './data/test'

#Saving the mapping of binary file to pickle
def save_class_mapping(name_to_indice_class):

    outcome_of_mapping = {val: key for key, val in name_to_indice_class.items()}
    with open("ResultsMap.pkl", 'wb') as file_write_stream:
        pickle.dump(outcome_of_mapping, file_write_stream)
    return outcome_of_mapping

# Function to create an instance of ImageDataGenerator with augmentation settings for generating more images with different shear_range and zoom_range
def dataGeneratorAsAugmentedImage():

    return ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)

#Loading and preprocessing the images
def image_loading_to_preprocess(path_of_located_image, size_of_targeted_image=(64, 64)):

    #resizing the image for further use
    processed_img = load_img(path_of_located_image, target_size = size_of_targeted_image)
    #converting images into numpy Array
    processed_img = img_to_array(processed_img)
    #Adding extra dimention
    processed_img = np.expand_dims(processed_img, axis=0)

    return processed_img

#Getting number of output neurons and class mapping
def get_class_mapping_and_output_neurons(data_generator):

    classes = data_generator.class_indices
    class_mapping = save_class_mapping(classes)
    output_neurons = len(class_mapping)

    return class_mapping, output_neurons

#Construction of neural network model that classifies images
def construct_compile_nn_model(output_neurons):
    
    #instance of Sequence
    new_mdl = Sequential()
    #adding maxpooling to model which is a downsampaling operation
    new_mdl.add(MaxPooling2D(pool_size=(2, 2)))
    #adding 2D convolution layer to model
    new_mdl.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    #reduce spatial dimensions
    new_mdl.add(MaxPooling2D(pool_size=(2, 2)))
    #flatten 2D layer to 1D vector
    new_mdl.add(Flatten())
    #relu activation function
    new_mdl.add(Dense(64, activation='relu'))
    #Multiclassification of produced possibility
    new_mdl.add(Dense(output_neurons, activation='softmax'))
    #compiling model and specifying loss function
    new_mdl.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    return new_mdl

#Function to train model
def training_model_with_generator(mdl_to_train, gen_data_flow_train, gen_data_flow_test):

    mdl_to_train.fit(
        gen_data_flow_train,
        steps_per_epoch=2,
        epochs=50,
        validation_data=gen_data_flow_test,
        validation_steps=10)

# Load the class mapping
def load_class_mapping():
    with open("ResultsMap.pkl", 'rb') as file_read_stream:
        return pickle.load(file_read_stream)

#Predict possible images using the trained model
def predict_with_trained_model(model, image_path, class_mapping):

    test_image = image_loading_to_preprocess(image_path)

    result = model.predict(test_image, verbose=0)

    predicted_class_id = np.argmax(result)

    return class_mapping[predicted_class_id]

#If this python program is run as main file
if __name__ == "__main__":

    # Load and preprocess images for training and testing
    train_datagen_aug_img = dataGeneratorAsAugmentedImage()
    test_datagen_aug_img = ImageDataGenerator()

    training_set_in_aug_img =  train_datagen_aug_img.flow_from_directory(
        #path of training image
        path_for_images_training,
        #Expected size for training data
        target_size=(64, 64),
        #number of images to be included
        batch_size=32,
        #defining categorical level for images
        class_mode='categorical')

    test_set_in_aug_img = test_datagen_aug_img.flow_from_directory(
        #path of testing image
        path_for_images_testing,
        #Expected size for testing data
        target_size=(64, 64),
        #number of images to be included
        batch_size=32,
        #defining categorical level for images
        class_mode='categorical')

    # To Get class_mapping and output_neurons
    class_mapping, output_neurons = get_class_mapping_and_output_neurons(training_set_in_aug_img)

    # Ready to compile and create model
    classifier = construct_compile_nn_model(output_neurons)

    # Training the model for prediction
    training_model_with_generator(classifier, training_set_in_aug_img, test_set_in_aug_img)

    # Predicting with trained model

    image_path_to_predict_1 = './predict_img/Img1.jpg'
    image_path_to_predict_2 = './predict_img/Img2.jpg'
    image_path_to_predict_3 = './predict_img/Img3.jpg'
    image_path_to_predict_4 = './predict_img/Img4.jpg'
    image_path_to_predict_5 = './predict_img/Img5.jpg'


    prediction1 = predict_with_trained_model(classifier, image_path_to_predict_1, class_mapping)
    prediction2 = predict_with_trained_model(classifier, image_path_to_predict_2, class_mapping)
    prediction3 = predict_with_trained_model(classifier, image_path_to_predict_3, class_mapping)
    prediction4 = predict_with_trained_model(classifier, image_path_to_predict_4, class_mapping)
    prediction5 = predict_with_trained_model(classifier, image_path_to_predict_5, class_mapping)


    #Final result of prediction
    print()
    print('Image shows the person :', prediction1)

    print()
    print('Image shows the person :', prediction2)

    print()
    print('Image shows the person :', prediction3)

    print()
    print('Image shows the person :', prediction4)

    print()
    print('Image shows the person :', prediction5)