import pandas as pd

import tensorflow as tf


def get_regicor_data(database):
    if database == 'CCA':
        """
        1-"EstudiDon": Is the subject identification (to find the corresponding image)
        2-"imtm_lcca_s”: Is the maximum value from IMT in left side of the CA
        3-"imtm_rcca_s": Is the maximum value from IMT in right side of the CA
        4-"imta_lcca_s": Is the mean value from IMT in left side of the CA
        5- "imta_rcca_s": Is the mean value from IMT in right side of the CA
        """
        imts_regicor = pd.read_csv('../datasets/CCA/REGICOR_4000/imt_data.csv', header=None)
        base_regicor_img_path = '../datasets/CCA/REGICOR_4000/CCA2_png'

    elif database == 'BULB':
        """
        1-"estudiparti": Is the subject identification (to find the corresponding image)
        2-"imtm_lbul_s”: Is the maximum value from IMT in left side of the CA
        3-"imtm_rbul_s": Is the maximum value from IMT in right side of the CA
        4-"imta_lbul_s": Is the mean value from IMT in left side of the CA
        5- "imta_rbul_s": Is the mean value from IMT in right side of the CA
        """
        imts_regicor = pd.read_csv('../datasets/BULB/REGICOR_3000/imt_dataBULB.csv', header=None)
        base_regicor_img_path = '../datasets/BULB/REGICOR_3000/BULB2_png'

    else:
        raise Exception('Database not recognized')

    imts_regicor.columns = ['EstudiDon', 'imtm_lcca_s', 'imtm_rcca_s', 'imta_lcca_s', 'imta_rcca_s']
    return imts_regicor, base_regicor_img_path


def check_gpu():
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found, training times on CPU could be to high')
    print('Found GPU at: {}'.format(device_name))
