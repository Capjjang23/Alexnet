import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense , Conv2D , Dropout , Flatten , Activation, Lambda, ZeroPadding2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
from bayes_opt import BayesianOptimization

# Define the input shape and number of classes
IMAGE_SIZE = 227
COLOR = 3
CLASSES = 26

# Define the AlexNet architecture
def create_alexnet(in_shape=(IMAGE_SIZE, IMAGE_SIZE, COLOR), n_classes=CLASSES, kernel_regular=regularizers.l2(l2=0.0005), optimizer='adam'):
    input_tensor = Input(shape=in_shape)

    # Layer 1
    x = Lambda(lambda image: tf.image.per_image_standardization(image))(input_tensor)
    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(11, 11), strides=(1, 1))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Layer 2
    x = ZeroPadding2D(padding=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), kernel_regularizer=kernel_regular)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 3
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=kernel_regular)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Layer 4
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Layer 5
    x = Dense(units=30, activation='relu')(x)
    x = Dense(units=n_classes, activation='softmax')(x)

    # Model compilation and summary
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    return model


# Define the function to be optimized by Bayesian optimization
def build_alexnet(learning_rate, l2_lambda):
    model = create_alexnet(
        in_shape=(227, 227, 3),
        n_classes=26,
        kernel_regular=regularizers.l2(l2=l2_lambda),
        optimizer=Adam(learning_rate=learning_rate))

    return model


# Define the search space
pbounds = {'learning_rate': (1e-5, 1e-2), 'l2_lambda': (1e-7, 1e-4)}


# Define Bayesian optimization object
bayes_opt = BayesianOptimization(
    f=build_alexnet,
    pbounds=pbounds,
    random_state=42)

# Perform Bayesian optimization
bayes_opt.maximize(init_points=10, n_iter=50, acq='ucb', kappa=2.576)

# Print the optimal hyperparameters
print('Result: ', bayes_opt.max)
