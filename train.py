import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import argparse, time, numpy as np

from torch.utils.data import random_split, DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision import datasets
from torch.optim import Optimizer, Adam
import torch
from torch import nn

from network.network import Classifier
from nominal_hyperparameters import Hyperparameters as hp

# test_data = datasets.KMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=transform,
# )

class Trainer():

    def get_mean_std(loader: DataLoader) -> tuple[float, float]:
        iterator = iter(loader)
        batch, _ = next(iterator)
        
        mean, std = np.zeros((batch.shape[1])), np.zeros((batch.shape[1]))

        for image_batch, _ in loader:

            mean += image_batch.mean(dim=(0,2,3)).numpy()
            std += image_batch.std(dim=(0,2,3)).numpy()
        
        mean = np.divide(mean, len(loader))
        std = np.divide(std, len(loader))
        return (mean.tolist(), std.tolist())

    def train(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer: Optimizer, device: torch.device) -> float:
        
        size, num_batches = len(dataloader.dataset), len(dataloader)
        epoch_train_loss, accuracy = 0, 0  

        model.train()

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

            # print loss & accuracy every 100 batches
            if batch % 100 == 0:
                loss, current = loss.item(), (batch+1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
        
        epoch_train_loss /= num_batches
        accuracy /= size
        print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {epoch_train_loss:>8f}\n")
        return (accuracy, epoch_train_loss)

    def validate(dataloader: DataLoader, model: nn.Module, loss_fn, device: torch.device) -> tuple[float, float]:
        
        size, num_batches = len(dataloader.dataset), len(dataloader)
        val_loss, accuracy = 0, 0  
    
        model.eval()
        
        with torch.no_grad():
            
            for (X, y) in dataloader:

                X, y = X.to(device), y.to(device) # send input to device
                pred = model(X) # forward pass

                loss = loss_fn(pred, y) # compute loss

                val_loss += loss.item() # accumulate loss 
                accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        val_loss /= num_batches
        accuracy /= size
        
        print(f"Validation Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {val_loss:>8f}\n")
        return (accuracy, val_loss)

    def plot_results(history: dict, path: str) -> None:

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(history["train_loss"], label="train_loss")
        plt.plot(history["val_loss"], label="val_loss")
        plt.plot(history["train_acc"], label="train_acc")
        plt.plot(history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(path)

    def save_model(model: nn.Module, path: str) -> None:
        torch.save(obj=model, f=path)

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-lr", "--learning_rate", type=float, default=hp.INIT_LR, help="learning rate")
    ap.add_argument("-bs", "--batch_size", type=int, default=hp.BATCH_SIZE, help="batch size")
    ap.add_argument("-e", "--epochs", type=int, default=hp.EPOCHS, help="number of epochs")
    ap.add_argument("-m", "--model", type=str, required=True, help="path to output trained model")
    ap.add_argument("-p", "--plot", type=str, required=True, help="path to output loss/accuracy plot")
    args = vars(ap.parse_args())  
    
    # looking for gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initializing trainer object
    trainer = Trainer()

    # The training data is first normalized so it has unit variance and zero mean
    print(f'[INFO] Loading training set and performing standardization...')
    training_data = datasets.KMNIST(root="data", train=True, download=True, transform=ToTensor())
    train_dataloader = DataLoader(training_data, batch_size=hp.BATCH_SIZE, shuffle=True)
    mean, std = trainer.get_mean_std(train_dataloader)
    print(f"Mean: {mean}, Std: {std}")

    transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    training_data = datasets.KMNIST(root="data", train=True, download=True, transform=transform)

    # train and validation dataset split and dataloader creation
    print("[INFO] generating the train/validation split...")
    training_data_size = int(len(training_data) * hp.TRAIN_SPLIT)
    val_data_size = int(len(training_data) * hp.VAL_SPLIT)
    (training_data, val_data) = random_split(training_data, [training_data_size, val_data_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(training_data, batch_size=hp.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=hp.BATCH_SIZE)
    # test_dataloader = DataLoader(test_data, batch_size=hp.BATCH_SIZE)

    # model, loss function and optimization strategy definition
    model = Classifier(numChannels=1, numClasses=len(training_data.dataset.classes)).to(device)
    loss_fn = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=hp.INIT_LR, weight_decay=0.003)

    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    start = time.time()

    for t in range(args["epochs"]):

        print(f"Epoch {t+1}/{args["epochs"]}\n-------------------------------")
        (train_acc, train_loss) = trainer.train(dataloader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
        (val_acc, val_loss) = trainer.validate(dataloader=val_dataloader, model=model, loss_fn=loss_fn, device=device)

        H["train_loss"].append(train_loss)
        H["train_acc"].append(train_acc)
        H["val_loss"].append(val_loss)
        H["val_acc"].append(val_acc)
        
    trainer.plot_results(history=H, path=args["plot"])
    trainer.save_model(model=model, path=args["model"])

    end = time.time()
    print(f"Done, Training time: {end-start} s")
    
if __name__ == "__main__":
    main()
    import pdb; pdb.set_trace()

