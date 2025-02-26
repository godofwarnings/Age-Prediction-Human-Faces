import torch
import pandas as pd
from tqdm import tqdm
from src.model import AgeModel
from src.dataset import AgeDataset

def predict(loader, model, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for img in tqdm(loader):
            img = img.to(device)
            pred = model(img)
            predictions.extend(pred.flatten().cpu().tolist())

    return predictions

if __name__ == "__main__":
    model_name = 'convnext_small'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AgeModel(model=model_name)
    model.load_state_dict(torch.load(f'models/Age_Prediction_{model_name}.pth'))
    model.to(device)

    test_path = 'data/test'
    test_ann = 'data/submission.csv'
    test_dataset = AgeDataset(test_path, test_ann, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=10)

    preds = predict(test_loader, model, device)

    submit = pd.read_csv('data/submission.csv')
    submit['age'] = preds
    submit.to_csv(f'results/predictions_{model_name}.csv', index=False)

    print(f"Predictions saved to results/predictions_{model_name}.csv")