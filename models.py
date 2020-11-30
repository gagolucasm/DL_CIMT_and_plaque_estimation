from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

import config


def get_imt_prediction_model():
    input_dim = 2 if config.INPUT_TYPE == 'img_and_mask' else 1
    input_image = Input(shape=(config.INPUT_SHAPE[0], config.INPUT_SHAPE[1], input_dim), name='input_image')
    base_model = Conv2D(32, (3, 3), activation='relu')(input_image)
    base_model = Conv2D(32, (3, 3), activation='relu')(base_model)
    base_model = MaxPooling2D(pool_size=(2, 2))(base_model)

    base_model = Conv2D(64, (3, 3), activation='relu')(base_model)
    base_model = MaxPooling2D(pool_size=(2, 2))(base_model)

    base_model = Conv2D(64, (3, 3), activation='relu')(base_model)
    base_model = MaxPooling2D(pool_size=(2, 2))(base_model)

    base_model = Flatten()(base_model)
    base_model = Dense(64, activation='relu')(base_model)
    base_model = BatchNormalization()(base_model)
    base_model = Dropout(rate=config.DROPOUT_RATE)(base_model)
    base_model = Dense(32, activation='relu')(base_model)
    base_model = BatchNormalization()(base_model)

    outputs = []
    if config.TARGET_COLUMNS['imt_max']['predict']:
        max_imt = Dense(32, activation='relu')(base_model)
        max_imt = Dropout(rate=config.DROPOUT_RATE)(max_imt)
        max_imt = Dense(16, activation='relu')(max_imt)
        max_imt = Dropout(rate=config.DROPOUT_RATE)(max_imt)
        max_imt = Dense(8, activation='relu')(max_imt)
        max_imt = Dense(1)(max_imt)
        max_imt = Activation('relu', name="max_imt", dtype='float32')(
            max_imt)  # For numerical stability in mixed precision
        outputs.append(max_imt)
    if config.TARGET_COLUMNS['imt_avg']['predict']:
        avg_imt = Dense(32, activation='relu')(base_model)
        avg_imt = Dropout(rate=config.DROPOUT_RATE)(avg_imt)
        avg_imt = Dense(16, activation='relu')(avg_imt)
        avg_imt = Dropout(rate=config.DROPOUT_RATE)(avg_imt)
        avg_imt = Dense(8, activation='relu')(avg_imt)
        avg_imt = Dense(1)(avg_imt)
        avg_imt = Activation('relu', name="avg_imt", dtype='float32')(
            avg_imt)  # For numerical stability in mixed precision
        outputs.append(avg_imt)

    if config.TARGET_COLUMNS['plaque']['predict']:
        # concatenation = concatenate([max_imt, avg_imt, base_model])
        # plaque = Dense(32, activation='relu')(concatenation)
        plaque = Dense(32, activation='relu')(base_model)
        plaque = Dropout(0.2)(plaque)
        plaque = Dense(16, activation='relu')(plaque)
        plaque = Dense(1)(plaque)
        plaque = Activation('sigmoid', name="plaque", dtype='float32')(
            plaque)  # For numerical stability in mixed precision
        outputs.append(plaque)

    model = Model(inputs=input_image, outputs=outputs)
    if config.DEBUG:
        print(model.summary())
    return model


if __name__ == '__main__':
    model = get_imt_prediction_model()
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)