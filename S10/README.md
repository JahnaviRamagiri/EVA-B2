**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)



To run the model, first we need to install the package 'pynoob' using !pip install pynoob.

Alternately, we can upload all the necessary packagesto the colab directory. The packages can be found in the S10/packages folder of EVA4 repo.

Link to pynoob: (https://github.com/abksyed/pynoob)

Link to model used: (https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S10/packages/CusResNet.py)


Objective
Achieve the following on the CIFAR-10 dataset:

Make sure to Add CutOut to your code. It should come from your transformations (albumentations)

Use this repo: https://github.com/davidtvs/pytorch-lr-finder (Links to an external site.)

Move LR Finder code to your modules

Implement LR Finder (for SGD, not for ADAM)

Implement ReduceLROnPlateau: https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau (Links to an external site.)

Find best LR to train your model

Use SDG with Momentum

Train for 50 Epochs.

Show Training and Test Accuracy curves

Target 88% Accuracy.

Run GradCAM on the any 25 misclassified images. Make sure you mention what is the prediction and what was the ground truth label.


Model Statistics:
Custom ResNet18 - BasicBlock - [2,2,2,2] - Last Layer Stride = 1 - So Final Output is 8 x 8!
Batch Size: 128
Number of Parameters: 11,173,962
Epochs: 50
Results
Achieved accuracy of

Test - 92.44%

Train - 96.51%

Viewing Data:

DataView1 DataView2 DataView3 DataView4 DataView5

Learning Rate Finder:

![LRFinderPlot](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S10/Results/findLR.png)

Best LR: 0.043287612810830614

Change in Learning Rate:
![ChangeLR](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S10/Results/changeLR.png)




Train and Test Accuracies and Loss:
![Test-Train Accuracy and Loss](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S10/Results/Trair_test_acc_loss.png)



Train vs Test Accuracy:
![Test-vs-Train Accuracy](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S10/Results/Train_vs_test.png)



Misclassified Images:

![MissClassifiedImages](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S10/Results/missclass.png)

MissClassifiedImages.png

Entire GradCAM for Mis Classified Images (w.r.t Predicted Class):

![MisClass_Pred](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S10/Results/mis_pred.png)

Entire GradCAM for Mis Classified Images(w.r.t Actual Class):

![MisClass_acc](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S10/Results/mis_act.png)


Class Wise Accuracies:
Accuracy of plane : 92 %

Accuracy of   car : 96 %

Accuracy of  bird : 88 %

Accuracy of   cat : 84 %

Accuracy of  deer : 93 %

Accuracy of   dog : 87 %

Accuracy of  frog : 95 %

Accuracy of horse : 95 %

Accuracy of  ship : 96 %

Accuracy of truck : 95 %
