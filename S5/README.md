**Objective**

1.	Build CNN Architecture to train MNIST Dataset
2.  99.4% (consistently shown in last few epochs, and not a one-time achievement)
3.	Less than or equal to 15 Epochs
4.	Less than 10000 Parameters
5.	Shown in minimum 5 steps


**Exp 1**
Target: To build Basic Model without Batchnorm, Dropout, Gap layers
1.	Get the set-up right
2.	Set Transforms
3.	Set Data Loader
4.	Set Basic Working Code
5.	Set Basic Training  & Test Loop

Batch size: 128
Lr=0.1

Result:
1.	Total Parameters: 13160
2.	Best Training Accuracy: 99.17
3.	Best Testing Accuracy: 99.18

Analysis:
1.	Parameters are higher than 10k
2.	Very less under fitting : Testing Accuracy = Training Accuracy which is implies a good model
3.	When trained for more epochs, the model would reach 99.40
4.	Parameters need to be reduced.
5.	Efficiency is to be increased




**Exp 2** 
Target: To increase efficiency using Batchnorm and Dropout
1.	Add Batchnorm
2.	Add dropout = 0.1
3.	Basic model remains the same 
Batch size: 128
Lr=0.1

Results:
1.	Parameters: 13160
2.	Best Training Accuracy: 99.19
3.	Best Test Accuracy: 99.29

Analysis:
1.	Efficiency has increased: Adding Dropout and batchnorm has worked.
2.	Under fitting in model expected as training includes dropout while testing doesn’t.
3.	The testing accuracy further needs to be increased.
4.	Parameters used need to be reduced – Focus for the next experiment




**Exp 3** 
Target: To Reduce Parameters using GAP
1.	Add GAP, slight changes made to the model
2.	Reduce batch size as the no. of parameters are assumed to be reduced on usage of GAP

Results:
1.	Parameters: 9752
2.	Best Training Accuracy: 
•	99.16 	(GAP and batch size =128)  	
•	99.28 (GAP and Batchsize=64) (19th epoch)

3.	Best Test Accuracy: 
•	99.35 (GAP and batch size = 128) 	
•	99.47 (GAP and Batchsize=64) (19th epoch)

Analysis:
1.	GAP layer of 8 kernels is used. If we reduce the kernels, more would be the convolutions and thus there is a chance to increase the accuracies but at the cost of increasing parameters.
2.	Adding GAP layer has decreased no. of parameters and yet accuracy increased : GAP is working
3.	Parameters have come below 10k: Parameter target reached. (Reducing kernels in GAP would increase parameters and thus, let us stick to this size.)
4.	Accuracy of 99.47 could be hit (though not consistent) at the 19th epoch (though not in less than 15 epochs) : 
5.	Reducing the Batchsize resulted in more iterations and thus has further increased the accuracy: Reducing batch size worked. Further reducing batchsize could be experimented with.



**Exp 4** 
Target: To further increase efficiency and create a consistent model
1.	Add image augmentation 
1.1.	 Image rotation by -7 to 7 degrees
1.2.	 Color Jitter
2.	Further Reduce Batch Size. (Batch size = 32)

Results:
1.	Parameters: 9752
2.	Best Training Accuracy: 
•	99.00 (image augmentation and batch size = 64) (17th epoch)  
•	98.94 (image augmentation and batch size = 32) (13th epoch)
•	99.06 (image augmentation and batch size = 32) (20th epoch)
3.	Best Test Accuracy: 
•	99.45 (image augmentation and batch size = 64) (17th epoch) 
•	99.49 (image augmentation and batch size = 32) (13th epoch)
•	99.53 (image augmentation and batch size = 32) (20th epoch)

Analysis:
1.	When only augmentation was performed, 9.4 or more was hit frequently but taking 20 epochs. Learning rate could experimented with to get the same result in lesser epochs.
2.	Model is under fitting as expected.
3.	Reducing batch size resulted in an accuracy of 99.49 at the 13th epoch.
4.	99.4 was hit 3 times before 15 epochs but there was a dip to 99.29 on the 15th epoch.
5.	The model can be made more consistent.




**Exp 5:** 
Target: To get the desired accuracy in lesser no. of epochs  and create a consistent model
1.	Change learning rate to achieve desired accuracy in lesser epochs
2.	Introduce LR scheduler (lr=0.02, step = 4, gamma = 0.5)

Results:
1.	Parameters: 9752
2.	Best Training Accuracy: 99.16 (13th epoch)
3.	Best Test Accuracy: 99.49 (13th epoch)

Analysis:
1.	99.4 Or above is hit 8 times before 15th epoch and 12 times before 20 epochs.
2.	Using lr of 0.2 and reducing it by half for every 4th epoch has helped us reach desired accuracy (99.40) more quickly and more consistently.
3.	Though Consistent, the accuracies did not exceed 99.49, thus a better LR scheduling is possible.
4.	Target of attaining a consistent accuracy of 99.4 or above, within 10k parameters, within 15 epochs is reached.
5.	Step is introduced manually and thus can go wrong at any point.
6.	Finding correct step can result in even better accuracies


