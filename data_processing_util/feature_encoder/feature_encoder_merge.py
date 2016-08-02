# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-02'
    Email:   '383287471@qq.com'
    Describe: 将多种 feature_encoder 融合
"""


class FeatureEncoderMerge(object):
    def __init__(self,
                 **feature_encoders
                 ):
        self.feature_encoders = feature_encoders

    def fit_transform(self, train_data=None):
        train_x_feature = []
        for encoder in self.feature_encoders.values():
            train_x_feature.append(encoder.fit_transform(train_data))
        return train_x_feature

    def transform(self, data):
        train_x_feature = []
        for encoder in self.feature_encoders.values():
            train_x_feature.append(encoder.transform(data))
        return train_x_feature

    def get_feature_encoder_by_name(self, name):
        return self.feature_encoders[name]

    def print_model_descibe(self):
        import pprint
        for encoder in self.feature_encoders.values():
            encoder.print_model_descibe()
