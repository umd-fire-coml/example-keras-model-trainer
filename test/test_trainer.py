import pytest
from src.trainer import MyDataset, my_model, train_model


@pytest.fixture
def my_dataset():
    return MyDataset.__init__()
    # return your dataset object


@pytest.fixture
def my_model():
    return my_model()
    # return your model object

# create the trainer tests below


def train_model():
    model = my_model()
    train_model(model)
