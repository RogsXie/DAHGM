import numpy as np
import torch


def calculate_metrics(confusion_matrix, total, correct):
    #Overall Accuracy (OA)
    OA = correct / total

    #Average Accuracy (AA)
    class_total = confusion_matrix.sum(axis=1)
    class_correct = np.diag(confusion_matrix)
    class_accuracy = np.divide(class_correct, class_total, where=class_total != 0)
    AA = np.mean(class_accuracy) * 100

    #Kappa
    Pe = np.sum(np.sum(confusion_matrix, axis=0) * np.sum(confusion_matrix, axis=1)) / (total ** 2)
    Kappa = (OA - Pe) / (1 - Pe)

    return OA * 100, AA, 100 * Kappa, class_accuracy



def evaluatethird(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for h11, l12, lg13, h21, l22, l23, h31, l32, lg33, labels in test_loader:
            h11 = h11.to(device)
            l12 = l12.to(device)
            lg13 = lg13.to(device)
            h21 = h21.to(device)
            l22 = l22.to(device)
            l23 = l23.to(device)
            h31 = h31.to(device)
            l32 = l32.to(device)
            lg33 = lg33.to(device)
            labels = labels.to(device)
            outputs = model(h11, l12, lg13, h21, l22, l23, h31, l32, lg33)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def evaluatetwo(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for h11, l12, labels in test_loader:
            h11 = h11.to(device)
            l12 = l12.to(device)
            labels = labels.to(device)
            outputs = model(h11, l12, )
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy