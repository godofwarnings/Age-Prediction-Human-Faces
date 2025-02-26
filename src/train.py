import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from model import AgeModel
import torch.optim as optim
from dataset import AgeDataset
from torch.utils.data import DataLoader

def train_model(model_name, lr, weight_decay):
    wandb.init(project='Age_Prediction', name=f'{model_name}', config={"model": model_name, "lr": lr})

    train_path = 'data/train'
    train_ann = 'data/train.csv'
    train_dataset = AgeDataset(train_path, train_ann, train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)

    model = AgeModel(model=model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)

    num_epochs = 100
    min_loss = float('inf')
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        wandb.log({"epoch": epoch})
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader, 0), desc="Batches", total=len(train_loader)):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward() 
            optimizer.step()
            running_loss += loss.item()

        wandb.log({"loss": running_loss/len(train_loader), "epoch": epoch})

        if running_loss < min_loss:
            min_loss = running_loss
            torch.save(model.state_dict(), f'models/Age_Prediction_{model_name}.pth')

        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    wandb.finish()

if __name__ == "__main__":
    models_arr = ['swin_v2_s', 'swin_v2_t', 'swin_v2_b', 'convnext_small', 'convnext_tiny', 'vit_b_16', 'resnet18', 'resnet34', 'resnet50', 'resnet101']
    weight_decays = [1e-4, 1e-3]
    lrs = [1e-4, 1e-3, 1e-2]

    for lr in lrs:
        for model_name in models_arr:
            for weight_decay in weight_decays:
                train_model(model_name, lr, weight_decay)