import pytest
from src.trainer import MyDataset, my_model

@pytest.fixture
def my_dataset():
    return None
    # return your dataset object


@pytest.fixture
def my_model():
    return None
    # return your model object

# create the trainer tests below