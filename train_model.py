from sugar_nets.data.sugar_dataset import SugarsDataset
from sugar_nets.models.sugar_classifier import FirstNetwork
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch
from cprint import cprint

def one_step_forward_backward():
    dataset = SugarsDataset(dataset_path="datasets/synthetic_data.csv")
    input_0, output_0 = dataset[15]
    

    model = FirstNetwork()
    optimizer = AdamW(params=model.parameters(), lr=0.0001)
    loss_func = CrossEntropyLoss()

    prediction_0 = model.forward(input_0)

    loss = loss_func(prediction_0, output_0.unsqueeze(0))

    loss.backward()
    optimizer.step()

def train_first_network():
    dataset = SugarsDataset(dataset_path="datasets/synthetic_data.csv")

    batch_size = 4096
    device_name = "cuda:0"

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = int(0.1 * len(dataset))

    train_dataset = torch.utils.data.Subset(dataset, indices=range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, indices=range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, indices=range(train_size + val_size, train_size + val_size + test_size))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = FirstNetwork()
    model = model.to(device=device_name)

    optimizer = AdamW(params=model.parameters(), lr=0.001)
    loss_func = CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10)

    epoch = 10

    for e in range(epoch):
        model.train()

        for b, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs, labels = batch
            
            inputs, labels = inputs.to(device=device_name), labels.to(device=device_name)

            predictions = model.forward(inputs)

            loss = loss_func(predictions, labels)

            loss.backward()
            optimizer.step()

            

            cprint.info(f"Epoch: {e+1}, Batch: {b+1}, Training Loss: {loss.item()}")
        
        scheduler.step() 

        model.eval()

        for b, batch in enumerate(val_dataloader):
            inputs, labels = batch
            predictions = model.forward(inputs)
            loss = loss_func(predictions, labels)
            cprint.info(f"Epoch: {e+1}, Batch: {b+1}, Validation Loss: {loss.item()}")

        torch.save(model.state_dict(), f"weights/sugar_nets_epoch_{e+1}.pth")

    for b, batch in enumerate(test_dataloader):
        inputs, labels = batch
        predictions = model.forward(inputs)
        loss = loss_func(predictions, labels)
        cprint.info(f"Test Batch: {b+1}, Test Loss: {loss.item()}")

if __name__ == "__main__":
    train_first_network()
