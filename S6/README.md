**Members:**

[Jahnavi Ramagiri](https://canvas.instructure.com/courses/1804302/users/25685093)

[Sachin Sharma](https://canvas.instructure.com/courses/1804302/users/23724529)

[Madalasa Venkataraman](https://canvas.instructure.com/courses/1804302/users/25685106)

[Syed Abdul Khader](https://canvas.instructure.com/courses/1804302/users/25685109)

Colab file:[https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S6/EVA4S6.ipynb](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S6/EVA4S6.ipynb)

To run the model, we need to upload all the necessary packagesto the colab directory. The packages can be found in the S6 folder of EVA4 repo.

**OBJECTIVE:**

Run your model for 25 epochs for each:

1. without L1/L2 with BN
2. without L1/L2 with GBN
3. with L1 with BN
4. with L1 with GBN
5. with L2 with BN
6. with L2 with GBN
7. with L1 and L2 with BN
8. with L1 and L2 with GBN

**Finding the Optimal Lambda Values for L1 and L2 Regularization:**

_For L1:_

Values checked for : 1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05

https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S6/Results/L1Lambda.PNG

The value of 1e-5, gave the best validation accuracy.

_For L2:_

Values checked for : 1e-5, 5e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05

https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S6/Results/L2Lambda.PNG

Here too the Lambda value of 1e-5 gave the best validation accuracy.

Hence, in the model we used the lambda value of 1e-5 (0.00001) for both L1 and L2.

**Validation Accuracy curves for all 8 task:**  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S6/Results/validation%20accuracy.png

**Validation Loss curves for all 8 task:** 
https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S6/Results/validation%20loss.png

**Misclassified Images for With L1/L2 with BN:**

https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S6/Results/NoL1_NoL2_BN_misclass.png

**Misclassified Images for With L1/L2 with GBN:**

https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S6/Results/NoL1_NoL2_GBN_misclass.png
