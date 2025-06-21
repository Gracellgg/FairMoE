import argparse, os
import shutil
from sklearn import metrics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data_loader.fitzpatrick17k_data import fitzpatrick17k_dataloader_score
from data_loader.ISIC2019_data import ISIC2019_dataloader_score
from fairness_metrics import compute_fairness_metrics
from torchvision import models
from sklearn.metrics import precision_recall_fscore_support

parser = argparse.ArgumentParser(description='MoE for fairness')
parser.add_argument('-n', '--num_classes', type=int, default=9,
                    help="number of classes; used for fitzpatrick17k")
parser.add_argument('-f', '--fair_attr', type=str, default="age",
                    help="fairness attribute; now support: gender, age; used for ISIC2019")
parser.add_argument('-d', '--dataset', type=str, default="fitzpatrick17k",
                    help="the dataset to use; now support: fitzpatrick17k, ISIC2019")
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--csv_file_name', type=str, default="D:/PycharmProjects/dataset/fitzpatrick17k/fitzpatrick17k.csv",
                    help="CSV file position; used for fitzpatrick17k and ISIC2019")
parser.add_argument('--image_dir', type=str, default="D:/PycharmProjects/dataset/fitzpatrick17k/dataset_images",
                    help="Image files directory; used for fitzpatrick17k and ISIC2019")
parser.add_argument('--root', type=str, default="D:/PycharmProjects/dataset/fitzpatrick17k/",
                    help="root of the dataset")
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of epochs in each cycle')
parser.add_argument('--save_dir', type=str, default="save",
                    help="directory to save models")
parser.add_argument('--pretrained_model', type=str, default=None,
                    help="path to pretrained model weights")
parser.add_argument('--model_suffix', type=str, default="val200epoch",
                    help="suffix for saved model filename")
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

def cal_metrics(label_list, y_pred_list, fair_binary_list):
    return {
        'skin_color/light_acc': metrics.accuracy_score(label_list[fair_binary_list == 0],
                                                       y_pred_list[fair_binary_list == 0]),
        'skin_color/dark_acc': metrics.accuracy_score(label_list[fair_binary_list == 1],
                                                      y_pred_list[fair_binary_list == 1]),
        'skin_color/light_precision': metrics.precision_score(label_list[fair_binary_list == 0],
                                                              y_pred_list[fair_binary_list == 0],
                                                              average='macro',
                                                              zero_division=0),
        'skin_color/dark_precision': metrics.precision_score(label_list[fair_binary_list == 1],
                                                             y_pred_list[fair_binary_list == 1],
                                                             average='macro',
                                                             zero_division=0),
        'skin_color/light_recall': metrics.recall_score(label_list[fair_binary_list == 0],
                                                        y_pred_list[fair_binary_list == 0], average='macro',
                                                        zero_division=0),
        'skin_color/dark_recall': metrics.recall_score(label_list[fair_binary_list == 1],
                                                       y_pred_list[fair_binary_list == 1], average='macro',
                                                       zero_division=0),
        'skin_color/light_f1_score': metrics.f1_score(label_list[fair_binary_list == 0],
                                                      y_pred_list[fair_binary_list == 0], average='macro',
                                                      zero_division=0),
        'skin_color/dark_f1_score': metrics.f1_score(label_list[fair_binary_list == 1],
                                                     y_pred_list[fair_binary_list == 1], average='macro',
                                                     zero_division=0),
    }

def test_model(n, testloader, ctype, f_attr, mode='val'):
    n.eval()  
    y_pred_list = []
    label_list = []
    fair_binary_list = []

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs = data["image"].float().to(device)
            labels = torch.from_numpy(np.asarray(data[ctype])).long().to(device)
            fair_binary = data[f_attr]
            if args.dataset == "ISIC2019":
                fair_binary = data['sex_color_binary'] if args.fair_attr == "gender" else data['age_approx_binary']
            else:
                fair_binary = data['skin_color_binary']

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = n(inputs)
            prediction = torch.argmax(outputs, dim=1)

            label_list.append(labels.detach().cpu().numpy())
            y_pred_list.append(prediction.detach().cpu().numpy())
            fair_binary_list.append(fair_binary.detach().cpu().numpy())

    label_list = np.concatenate(label_list)
    y_pred_list = np.concatenate(y_pred_list)
    fair_binary_list = np.concatenate(fair_binary_list)

    accuracy = np.sum(y_pred_list == label_list) / len(label_list)
    precision, recall, f1, _ = precision_recall_fscore_support(label_list, y_pred_list, average='macro')
    print(mode + f" Model Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    result = cal_metrics(label_list, y_pred_list, fair_binary_list)
    fairness_metrics = compute_fairness_metrics(label_list[fair_binary_list != -1],
                                                y_pred_list[fair_binary_list != -1],
                                                fair_binary_list[fair_binary_list != -1])
    for k, v in result.items():
        print(f'{k}:{v:.4f}')
    for k, v in fairness_metrics.items():
        print(f'{k}:{v:.4f}')

    average_accuracy = metrics.f1_score(label_list[fair_binary_list == 0],
                                                      y_pred_list[fair_binary_list == 0], average='macro',
                                                      zero_division=0)
    return average_accuracy

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create save directory if it doesn't exist
    os.makedirs(os.path.join(args.save_dir, args.dataset), exist_ok=True)

    if args.dataset == "fitzpatrick17k":
        num_classes = args.num_classes
        if num_classes == 3:
            ctype = "high"
        elif num_classes == 9:
            ctype = "mid"
        elif num_classes == 114:
            ctype = "low"
        else:
            raise NotImplementedError
        f_attr = "skin_color_binary"
        trainloader, lightloader, darkloader, skin_type_list_dataloader, valloader, testloader = fitzpatrick17k_dataloader_score(
            args.batch_size, args.workers, args.image_dir, args.csv_file_name, args.root, ctype)
    elif args.dataset == "ISIC2019":
        num_classes = 9
        ctype = "class_id"
        f_attr = "sex_color_binary" if args.fair_attr == "gender" else "age_approx_binary"
        trainloader, lightloader, darkloader, valloader, testloader = ISIC2019_dataloader_score(args.batch_size,
                                                                                                args.workers,
                                                                                                args.image_dir,
                                                                                                args.csv_file_name,
                                                                                                args.root,
                                                                                                args.fair_attr,
                                                                                                ctype)
    else:
        raise NotImplementedError

    print('data loaded')

    resnet18 = models.resnet18(pretrained=True)
    resnet18.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, args.num_classes)

    net = resnet18.to(device)

    # Load pretrained weights if specified
    if args.pretrained_model:
        print(f"Loading pretrained weights from {args.pretrained_model}")
        net.load_state_dict(torch.load(args.pretrained_model))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    net.train()

    best_model = net.state_dict()
    best_accuracy = 0.0
    for epoch in range(args.epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for i, data in enumerate(trainloader, 0):
            inputs = data['image']
            labels = data[ctype]
            fair_binary = data[f_attr]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())

            prediction = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(prediction == labels).item()
            total_samples += labels.size(0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(trainloader)
        accuracy = correct_predictions / total_samples
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, loss: {total_loss:.4f}, accuracy: {accuracy:.4f}")

        if epoch % 1 == 0:
            net.eval()

            accuracy = test_model(net, valloader, ctype, f_attr)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = net.state_dict()
                model_path = os.path.join(args.save_dir, args.dataset, f"resnet18_{f_attr}_{ctype}_{args.model_suffix}.pth")
                torch.save(best_model, model_path)
                print(f"Best model saved at Epoch {epoch} to {model_path}")

                test_acc = test_model(net, testloader, ctype, f_attr, mode='test')

            net.train()
