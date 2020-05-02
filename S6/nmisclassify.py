import numpy as np
import matplotlib.pyplot as plt


def nMostMisclassified(n, test_loader, model, device):
    scores = []
    misclassified_images = []
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        output = model(data)
        predictions = output.argmax(dim=1, keepdim=True)
        for image, score, preds, target in zip(data, output, predictions, targets):
            if(~preds.eq(target.view_as(preds))):
                preds = preds.cpu()
                target = target.cpu()
                image = image.cpu()
                score = score.cpu()
                scores.append(score[target])
                misclassified_images.append(image)
    sorted_images = [x for _, x in sorted(zip(scores, misclassified_images))]
    for image, score1 in zip(sorted_images[:n], sorted(scores)[:n]):
        plt.imshow(image.cpu().numpy().reshape((28, 28)))
        plt.show()
