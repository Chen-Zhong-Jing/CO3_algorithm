# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 22:47:26 2022

@author: Zhong_Jing
"""

from tensorflow import keras

def DenseNet121(input):
    return keras.applications.DenseNet121(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
    )


def DenseNet169(input):
    return keras.applications.DenseNet169(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
    )


def DenseNet201(input):
    return keras.applications.DenseNet201(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
    )


def ResNet50(input):
    return keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax",
    )


def ResNet101(input):
    return keras.applications.ResNet101(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax",
    )


def ResNet152(input):
    return keras.applications.ResNet152(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax",
    )


def ResNet50V2(input):
    return keras.applications.ResNet50V2(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax",
    )


def ResNet101V2(input):
    return keras.applications.ResNet101V2(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax",
    )


def ResNet152V2(input):
    return keras.applications.ResNet152V2(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax",
    )


def NASNetMobile(input):
    return keras.applications.NASNetMobile(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10
    )


def NASNetLarge(input):
    return keras.applications.NASNetLarge(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10
    )


def VGG16(input):
    return keras.applications.VGG16(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        classifier_activation="softmax"
    )


def VGG19(input):
    return keras.applications.VGG19(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=input.shape[1:],
        pooling=None,
        classes=10,
        #classifier_activation="softmax",
    )

build_model = {'DenseNet121': DenseNet121,
               'DenseNet169': DenseNet169,
               'DenseNet201': DenseNet201,
               'ResNet50': ResNet50,
               'ResNet50V2': ResNet50V2,
               'ResNet101': ResNet101,
               'ResNet101V2': ResNet101V2,
               'ResNet152': ResNet152,
               'ResNet152V2': ResNet152V2,
               'NASNetMobile': NASNetMobile,
               'NASNetLarge': NASNetLarge,
               'VGG16': VGG16,
               'VGG19': VGG19
               }