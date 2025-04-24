from data import DataLoaderMNIST
from model import MyModel


class Trainer:
    def __init__(self, model):
        self._model = model

    def __call__(self, train_dataset, valid_dataset, epochs):
        # TODO: Implement training loop
        pass


if __name__ == "__main__":
    model = MyModel("MNISTClassifier")

    data_loader = DataLoaderMNIST()
    train_dataset = data_loader.train_dataset
    valid_dataset = data_loader.valid_dataset

    train = Trainer(model)
    train(train_dataset, valid_dataset, epochs=20)

    model.save("my_model.keras")
