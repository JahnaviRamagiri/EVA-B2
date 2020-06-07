**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)



To run the model, we need to upload all the necessary packagesto the colab directory. The packages can be found in the S9 folder of EVA4 repo.




**Assignment:**

Move your last code's transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)

Please make sure that your test_transforms are simple and only using ToTensor and Normalize

Implement GradCam function as a module. 

Your final code (notebook file) must use imported functions to implement transformations and GradCam functionality

Target Accuracy is 87%

Submit answers to S9-Assignment-Solution. 



**Model Statistics:**

Custom ResNet18 - BasicBlock - [2,2,2,2] - Last Layer Stride = 1 - So Final Output is 8 x 8!

Batch Size: 128

Number of Parameters: 11,173,962

Epochs: 15


**Results**
Achieved accuracy of

Test - 86.95%

Train - 89.05%


Train and Test Accuracies and Loss:

![Test-Train Accuracy and Loss](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S9/Results/train_test_loss_accuracy.png)

Train vs Test Accuracy:

![Test-vs-Train Accuracy](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S9/Results/Train_vs_test.png)

Misclassified Images:

![MissClassifiedImages](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S9/Results/missclass.png)


GradCAM HeatMap for Mis Classified Images: 

![Mis_HeatMap](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S9/Results/Heatmap.png)

GradCAM for Mis Classified Images: 

![Mis_GradCAM](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S9/Results/gradcam.png)


**Class Wise Accuracies:**

Accuracy of plane : 85 %

Accuracy of car : 97 %

Accuracy of bird : 85 %

Accuracy of cat : 79 %

Accuracy of deer : 92 %

Accuracy of dog : 80 %

Accuracy of frog : 83 %

Accuracy of horse : 86 %

Accuracy of ship : 88 %

Accuracy of truck : 91 %



