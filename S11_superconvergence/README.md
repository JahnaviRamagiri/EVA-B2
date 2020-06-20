**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

**Assignment:**

1. Write a code that draws this curve (without the arrows). In submission, you&#39;ll upload your drawn curve and code for that

   ![curve](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S11_superconvergence/Results_Final/curve.png)
   
2. Write a code which
  1. uses this new ResNet Architecture for Cifar10:
    1. PrepLayer - Conv 3x3 s1, p1) \&gt;\&gt; BN \&gt;\&gt; RELU [64k]
    2. Layer1 -
      1. X = Conv 3x3 (s1, p1) \&gt;\&gt; MaxPool2D \&gt;\&gt; BN \&gt;\&gt; RELU [128k]
      2. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
      3. Add(X, R1)
    3. Layer 2 -
      1. Conv 3x3 [256k]
      2. MaxPooling2D
      3. BN
      4. ReLU
    4. Layer 3 -
      1. X = Conv 3x3 (s1, p1) \&gt;\&gt; MaxPool2D \&gt;\&gt; BN \&gt;\&gt; RELU [512k]
      2. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
      3. Add(X, R2)
    5. MaxPooling with Kernel Size 4
    6. FC Layer
    7. SoftMax
  2. Uses One Cycle Policy such that:
    1. Total Epochs = 24
    2. Max at Epoch = 5
    3. LRMIN = FIND
    4. LRMAX = FIND
    5. NO Annihilation
  3. Uses this transform -RandomCrop 32, 32 (after padding of 4) \&gt;\&gt; FlipLR \&gt;\&gt; Followed by CutOut(8, 8)
  4. Batch size = 512
  5. Target Accuracy: 90%.
  6. The lesser the modular your code is (i.e. more the code you have written in your Colab file), less marks you&#39;d get.
3. Questions asked are:
  1. Upload the code you used to draw your ZIGZAG or CYCLIC TRIANGLE plot.
  2. Upload your triangle Plot which was drawn with your code.
  3. Upload the link to your GitHub copy of Colab Code.
  4. Upload the github link for the model as described in A11.
  5. What is your test accuracy?

**Model Statistics:**

- Custom Model
- Batch Size: 512
- Number of Parameters: 6,573,130
- Epochs: 21


**Results**

Cyclic Curve: 

![Cyclic Curve](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S11_superconvergence/Results_Final/cyclicLR.png)

Average loss: -7.7352, Best Accuracy: 9153/10000 (91.53%) (23rd Epoch)


Learning Rate Finder:

![LR vs Loss](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S11_superconvergence/Results_Final/LRvsLoss.png)


Best loss 1.1508058843750388

Best LR:  0.0092968331284346

![LR vs Accuracy](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S11_superconvergence/Results_Final/LRvsAcc.png)

Best acc 66.2109375

Best LR:  0.01096065651201893

Cyclic Change in Learning Rate:

![ChangeLR](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S11_superconvergence/Results_Final/ChangeLR.png)

Train and Test Accuracies and Loss:

![Test-Train Accuracy and Loss](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S11_superconvergence/Results_Final/Train_test_Loss_Acc.png)

Train vs Test Accuracy:

![Test-vs-Train Accuracy](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S11_superconvergence/Results_Final/TrainvsTest.png)

Misclassified Images:

![MissClassifiedImages](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S11_superconvergence/Results_Final/misclass.png)


GradCAM wrt Predicted for Mis Classified Images: 

![Mis_Pred](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S11_superconvergence/Results_Final/Gradcam_pred.png)

GradCAM wrt Actual for Mis Classified Images: 

![Mis_act](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S11_superconvergence/Results_Final/Gradcam_acc.png)



**Class Wise Accuracies:**


Accuracy of plane : 95 %

Accuracy of   car : 97 %

Accuracy of  bird : 89 %

Accuracy of   cat : 82 %

Accuracy of  deer : 92 %

Accuracy of   dog : 85 %

Accuracy of  frog : 93 %

Accuracy of horse : 92 %

Accuracy of  ship : 94 %

Accuracy of truck : 90 %



