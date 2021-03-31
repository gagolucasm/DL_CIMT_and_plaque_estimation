#!/usr/bin/env python
# coding: utf-8

import predict_imt

if __name__ == '__main__':
    train = True  # Whether to train or not the models. If it is false, it will run the evaluation pipeline.
    # Define lists of parameters to test
    database_list = ['CCA']
    input_type_list = ['img_and_mask', 'mask', 'img']  # ['mask', 'img','img_and_mask']
    target_columns_list = ['imt_max', 'imt_avg', 'plaque']  # ['imt_max', 'imt_avg', 'plaque', 'all']
    input_shape = (470, 445)
    # Iterate over them training the models
    for database in database_list:
        for input_type in input_type_list:
            for target in target_columns_list:
                target_columns = {}
                for item in ['imt_max', 'imt_avg', 'plaque']:
                    target_columns[item] = {'predict': True if item == target or target == 'all' else False,
                                            'weight': 1 if item == 'plaque' else 1,
                                            'loss': 'binary_crossentropy' if item == 'plaque' else 'mean_squared_error'}

                predict_imt.train_imt_predictor(database=database, input_type=input_type, input_shape=input_shape,
                                                target_columns=target_columns, silent_mode=True, train_model=train)
