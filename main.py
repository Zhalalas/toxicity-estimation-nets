from data.sugars_dataset import SugarsDataset
from models.first_network import FirstNetwork
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from cprint import cprint

def one_step_forward_backward():
    dataset = SugarsDataset(dataset_path="datasets/synthetic_data.csv")
    input_0, output_0 = dataset[15]

    model = FirstNetwork()
    optimizer = AdamW(params=model.parameters(), lr=0.001)
    loss_func = CrossEntropyLoss()

    prediction_0 = model.forward(input_0)

    loss = loss_func(prediction_0, output_0.unsqueeze(0))

    loss.backward()
    optimizer.step()

def train_first_network():
    dataset = SugarsDataset(dataset_path="datasets/synthetic_data.csv")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = FirstNetwork()
    optimizer = AdamW(params=model.parameters(), lr=0.001)
    loss_func = CrossEntropyLoss()

    epoch = 10

    for e in range(epoch):
        for b, batch in enumerate(dataloader):
            optimizer.zero_grad()
            inputs, labels = batch

            predictions = model.forward(inputs)

            loss = loss_func(predictions, labels)

            loss.backward()
            optimizer.step()
            

            cprint.info(f"Epoch: {e+1}, Batch: {b+1}, Loss: {loss.item()}")



    torch.save(model.state_dict(), "weights/first_network.pth")

if __name__ == "__main__":
    train_first_network()
