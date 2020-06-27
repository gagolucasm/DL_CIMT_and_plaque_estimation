#!/usr/bin/env python
# coding: utf-8

import predict_imt

if __name__ == '__main__':
    # Define lists of parameters to test
    database_list = ['BULB', 'CCA']
    input_type_list = ['img', 'mask']
    target_columns_list = ['imt_max', 'imt_avg', 'plaque']
    input_shape = (512, 512)

    # Iterate over them training the models
    for database in database_list:
        for input_type in input_type_list:
            for target in target_columns_list:
                target_columns = {}
                for item in target_columns_list:
                    target_columns[item] = {'predict': True if item == target else False,
                                            'weight': 1.,
                                            'loss': 'binary_crossentropy' if item == 'plaque' else 'mean_squared_error'}

                predict_imt.train_imt_predictor(database=database, input_type=input_type, input_shape=input_shape,
                                                target_columns=target_columns)
