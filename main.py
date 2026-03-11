from training.train import train_model
from experiments.test_prediction import test_prediction


def main():
    # 1 Treinar a rede
    model, scale, sequence = train_model()

    # 2 Testar previsão
    test_prediction(model, scale, sequence)


if __name__ == "__main__":
    main()