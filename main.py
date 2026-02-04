from data.sugars_dataset import SugarsDataset


def train_first_network():
    dataset = SugarsDataset(dataset_path="datasets/synthetic_data.csv")
    print(dataset[0])
    return


if __name__ == "__main__":
    train_first_network()
