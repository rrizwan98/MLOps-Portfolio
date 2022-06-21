import json
import logging
import os
# import joblib
import pytest
from prediction_service.prediction import form_response, api_response
import prediction_service
import yaml

input_data = {
    "incorrect_range": 
    {"SepalLengthCm": 7897897, 
    "SepalWidthCm": 555, 
    "PetalLengthCm": 99, 
    "PetalWidthCm": 99
    },

    "correct_range":
    {"SepalLengthCm": 4.4, 
    "SepalWidthCm": 2.1, 
    "PetalLengthCm": 1.1, 
    "PetalWidthCm": 0.2 
    },

    "incorrect_col":
    {"Sepal Length Cm": 4.4, 
    "Sepal Width Cm": 2.1, 
    "Petal Length Cm": 1.1, 
    "Petal Width Cm": 0.2 
    }
}
# species range
TARGET_range = {
    "min": 0,
    "max": 2
}

def test_form_response_correct_range(data=input_data["correct_range"]):
    res = form_response(data)
    assert  TARGET_range["min"] <= res <= TARGET_range["max"]

def test_api_response_correct_range(data=input_data["correct_range"]):
    res = api_response(data)
    assert  TARGET_range["min"] <= res["response"] <= TARGET_range["max"]

def test_form_response_incorrect_range(data=input_data["incorrect_range"]):
    with pytest.raises(prediction_service.prediction.NotInRange):
        res = form_response(data)

def test_api_response_incorrect_range(data=input_data["incorrect_range"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInRange().message

def test_api_response_incorrect_col(data=input_data["incorrect_col"]):
    res = api_response(data)
    assert res["response"] == prediction_service.prediction.NotInCols().message