**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

Colab file:(https://colab.research.google.com/drive/1h8p58mbZn-TeWB_mSMdpKupKa63xxIAc?authuser=1#scrollTo=BjgKiK1-cswd)

To run the model, we need to upload all the necessary packagesto the colab directory. The packages can be found in the S8 folder of EVA4 repo.


### **Objective**

Achieve the following on the **CIFAR-10** dataset:

- Extract the ResNet18 model from this repository and add it to your API/repo
- Use data loader, train, test, and utils code to train ResNet18 on Cifar10.
- Use default ResNet18 code (so params are fixed).
- achieve **85%** accuracy, as many epochs as required.

### **Model Statistics:**

- ResNet18 - BasicBlock - [2,2,2,2]
- Batch Size: 128
- Number of Parameters: 11,173,962
- Epochs: 50

### **Results**

Achieved accuracy of

**Test - 91.71%**
**Train - 98.14%**

Misclassified Images:

![MissClassifiedImages](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S8/output/MissClassify.png)

Train and Test Accuracies and Loss:

![Test-Train Accuracy and Loss](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S8/output/LossandAcc.png)

Train vs Test Accuracy:

![Test-vs-Train Accuracy](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S8/output/TestvTrainAcc.png)

Class Wise Accuracies:
Accuracy of plane : 91 %

Accuracy of   car : 93 %

Accuracy of  bird : 91 %

Accuracy of   cat : 83 %

Accuracy of  deer : 94 %

Accuracy of   dog : 83 %

Accuracy of  frog : 94 %

Accuracy of horse : 92 %

Accuracy of  ship : 96 %

Accuracy of truck : 95 % 

