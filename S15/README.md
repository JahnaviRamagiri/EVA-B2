**Dataset Creation:**

You must have 100 background, 100x2 (including flip), and you randomly place the foreground on the background 20 times, you have in total 100x200x20 images.

In total you MUST have:

1. 400k fg\_bg images
2. 400k depth images
3. 400k mask images
4. generated from:
  1. 100 backgrounds
  2. 100 foregrounds, plus their flips
  3. 20 random placement on each background.
5. Now add a readme file on GitHub for Project 15A:
  1. Create this dataset and share a link to GDrive (publicly available to anyone) in this readme file.
  2. Add your dataset statistics:
    1. Kinds of images (fg, bg, fg\_bg, masks, depth)
    2. Total images of each kind
    3. The total size of the dataset
    4. Mean/STD values for your fg\_bg, masks and depth images
  3. Show your dataset the way I have shown above in this readme
  4. Explain how you created your dataset
    1. how were fg created with transparency
    2. how were masks created for fgs
    3. how did you overlay the fg over bg and created 20 variants
    4. how did you create your depth images?
6. Add the notebook file to your repo, one which you used to create this dataset
7. Add the notebook file to your repo, one which you used to calculate statistics for this dataset


# Data Preparation:

This part involved the prepararion of data, 1.2M images were prepared and their details  are below.


## GDrive link for the dataset

https://drive.google.com/drive/folders/1Fb0LeHB-km0M959_lyhbdF0Y52-id9bf?usp=sharing

## Dataset Statistics

## Kinds of images 

Image | Type | Channels | Dimension
----- | ---- | -------- | ---------
fg | png | 4 | 112*112
bg | jpg | 3 | 224*224
fg_bg | jpg | 3 | 224*224
fg_bg_masks | jpg | 1 | 224*224
depth | jpg | 1 | 224*224

## Total images of each kind
Image | Count
----- | -----
 fg | 100
 bg | 100
 fg_bg | 400000
 fg_bg_masks | 400000
 depth | 400000

## Total size of the dataset
4.8 GB

## Mean/STD values 
Image | Mean | STD
----- | ---- | ----
bg    | [0.5558092594146729, 0.5201340913772583, 0.463156521320343] | [0.2149990200996399, 0.21596555411815643, 0.23049025237560272]
fg_bg | [0.5455222129821777, 0.5086212158203125, 0.45718181133270264] | [0.22610004246234894, 0.2249932438135147, 0.23590309917926788]
fg_bg_masks | [0.05790501832962036] | [0.22068527340888977]
depth images | [0.40361160039901733]| [0.19922664761543274]

# Dataset Images

## Background
![BG](https://raw.githubusercontent.com/ganeshkcs/EVA4B2/master/S15A/Data_Sample/bg.png)

## Foreground
![FG](https://raw.githubusercontent.com/ganeshkcs/EVA4B2/master/S15A/Data_Sample/fg.png)

## Foreground Mask
![FG Mask](https://raw.githubusercontent.com/ganeshkcs/EVA4B2/master/S15A/Data_Sample/fg_mask.png)

## FG BG
![FG BG](https://raw.githubusercontent.com/ganeshkcs/EVA4B2/master/S15A/Data_Sample/fg_bg.png)

## FG BG Mask
![FG BG Mask](https://raw.githubusercontent.com/ganeshkcs/EVA4B2/master/S15A/Data_Sample/fg_bg_mask.png)

## Depth Images
![Depth](https://raw.githubusercontent.com/ganeshkcs/EVA4B2/master/S15A/Data_Sample/depth.png)

## How dataset was prepared?

* 100 background images were collected
* 100 foreground images were collected. Prefrerred white background and png with transparent background
* Foreground images with white background was made transparent, using GIMP tool. 
  Steps :
    * Open the image in GIMP
    * Select Fuzzy Select Tool and click on white area of the image
    * To add alpha channel - Click on Layer --> Transparency --> Add Alpha Channel 
    * To make white color as transparent - Click on Layer --> Transparency --> Color to alpha --> (Make sure white color is choose in pop-up) --> click OK
    * Now all the white area would have been converted to transparent. Save/Export the image
* Foreground mask was prepared by using opencv.  
    * Alpha (4th) channel of FG alone is created as separate 1 channel mask image. 
    * Code: https://github.com/ganeshkcs/EVA4B2/blob/master/S15A/S15_FG_MASK.ipynb
* **FG BG Preparation**
    * PIL was used
    * FG is overlaid on BG, at (x,y) of BG, using following code :
      ```
      # Creating black image to store the mask of bg_fg images
      black_imgage_for_fg_mask = Image.fromarray(black_image,mode='1')
      black_image_for_fg_flip_mask = Image.fromarray(black_image,mode='1')

      # Creating Flip images
      fg_flip_image = copy_for_fg.transpose(PIL.Image.FLIP_LEFT_RIGHT)
      fg_flip_mask_image = copy_for_fg_mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)

      # Overlaying images
      copy_for_bg_fg.paste(copy_for_fg, (x_cord,y_cord),copy_for_fg)
      copy_for_bg_flip_fg.paste(fg_flip_image, (x_cord,y_cord),fg_flip_image)

      black_imgage_for_fg_mask.paste(copy_for_fg_mask, (x_cord,y_cord), copy_for_fg_mask)
      black_image_for_fg_flip_mask.paste(fg_flip_mask_image, (x_cord,y_cord), fg_flip_mask_image)

       ```
    * For one BG, 
        * one FG is taken, and 20 random co-ordinates was generated with in the BG bounds
        * overlay is done with above code at each random co-ordinate, and 20 resulting images are saved
        * FG is then fliped, using PIL, and again the above 2 steps are repeated and 20 images are saved
        * The process is repeated for all FG. At the end, we had 4000 images generated for one BG.
    * Above step was repeated for all 100 BGs and we had 400K images ready
    * Files were written to zip, for easy access. 100 Zip files were created, where each zip file corresponds to 1 BG with 4000 images
    * Code : https://github.com/ganeshkcs/EVA4B2/blob/master/S15A/S15_ZIP_OVERLAY.ipynb
    
 * **Dense Depth Images Preparation**
    * When we ran the model for the fg_bg images, prediction was not good, since all our FGs were either subtle or blending with background.
    * Hence sharpened the FGs, before FG BG image dataset was prepared
    * Dense depth load_images code was modified to read images from Zip directly
    * Faced memory error when more images where passed, so prediction was done in chunks, taking 50 images and looping through them.
    * Wrote following code, for processing in chunks :
      ```
      for file_name in mylist[start:end]:
        print(file_name, start, end)
        print("Bg Loop",datetime.datetime.now())
        with zipfile.ZipFile(file_name["zip_file_name"], 'r') as zip:
          start = timeit.default_timer()
          print("start", start)
          file_list = zip.namelist()
          new_zip_name = file_name["bg_number"]
          dense_depth_zip = zipfile.ZipFile(output_dir+f'/fb_bg_depth{new_zip_name}.zip', mode='a', compression=zipfile.ZIP_STORED)
          for i in range(0, 4000, 50):
            snipped_list = file_list[i:i+50]
            inputs= load_zip_images(zip, snipped_list)
            inputs = scale_up(2, inputs)
            outputs = predict(model, inputs)
            save_output(outputs, output_dir, snipped_list, dense_depth_zip, is_rescale=True)
      
      stop = timeit.default_timer()
      execution_time = stop - start
      dense_depth_zip.close()
      print("Program Executed in "+str(execution_time))

       ```
    * Resulting output was converted to grey scale, since they save space and loading will be easier for the final assignment.
    * For one folder with 4000 images, it took aroung ~6 mins and so we happened to have patience for 10 long hours, to see the final dataset with 400K depth images
    * All the modifications done in dense depth can be found here : https://github.com/ganeshkcs/DenseDepth
    
 * **Dataset Statistics** 
      * Wrote a custom Zip dataset loader, using pytorch, to read the images
      * mean and std where calculated
      * Code : https://github.com/ganeshkcs/EVA4B2/blob/master/S15A/S15_Statistics.ipynb
      
  # S15 Implementation
  
  **Aim** : Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object.

To implement this, a model had to be chosen which can do this job with least possible parameters (to reduce memory requirement) and the highest possible accuracy.

After going through various models which included ResNEt18, DenseNEt among others, UNet was chosen.

UNET Architechture:

![](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Images/UNet%20Architecture.png)

UNet, evolved from the traditional convolutional neural network, was first designed and applied in 2015 to process biomedical images. As a general convolutional neural network focuses its task on image classification, where input is an image and output is one label, but in biomedical cases, it requires us not only to distinguish whether there is a disease, but also to localise the area of abnormality.

UNet is dedicated to solving this problem. The reason it is able to localise and distinguish borders is by doing classification on every pixel, so the input and output share the same size.

Inspired from the usage of Medical domain, as the requirement in our problem statement is also to produce images as output, I have chosen UNet as my base model.

Data Management: While creating the Dataset, as shown above, initially we had take png images for the dataset which happened to consume a lot of space. As a result, the overlay images, mask images all together resulted in a 20GB file.

To reduce this computation cost, we have decided to convert all the images except foreground (fg) into jpg format. FFMPEG was used for these manipulations.

The Background size was fixed to 224\*224\*3 as advised to, but we could play around with the foreground size. We experimented with 36\*36 pixel fg first, and found that this isn&#39;t giving good depth results. Then we switched to 96\*96. The output depth was much better for 96\*96 than 36\*36 but the best results were found with 112\*112. Thus, we stuck to 112\*112 as our fg size.

**DNN Model:**

Parameters – 86,31,810

Epochs – 10

Average Loss – Mask -0.65 Depth – 0.57

- As we have 2 inputs to give and 2 outputs to get from a single network, the inputs (bg, fg\_bg) were concatenated to make a total of 6 channels (3+3). Therefore, the input to the network was of the size 224\*224\*6, which was the transformed to 64\*64\*6 to reduce computational requirement.
- Architecture:

Input (64\*64) -\&gt; double conv (64\*64) -\&gt; Downsample (32\*32-16\*16-8\*8) -\&gt; downsample (4\*4) -\&gt; Upsample (8\*8 – 16\*16 -32\*32 -64\*64) -\&gt;output =\&gt;Mask (64\*64) and Depth (64\*64)

- The output, 64\*64, both depth and mask are obtained in grayscale.

Optimizer – Adam and RMSProp optimizers were tried with, in which Adam gave better results.

Scheduler – ReduceLROnPlateau – with a plateau of 2

Loss: Binary Cross Entropy (BCE) and structural similarity index measure (SSIM) were used to calculate the loss, BCE turned out to be better of the both.

Mask Loss - BCE

Depth Loss - BCE

**Challenges:**

- Accessing Dataset: The challenges faced while creating the dataset were numerous. Handling the Dataset and training the model with the custom dataset was the hardest part.
- While I was trying to attach the dataset to the model, I/O issues were getting raised in colab.
- A folder with more than 10000 images would most likely crash
- To solve this issue we created 100 sets of fg\_bg, fg\_bg\_mask, fg\_bg\_depth each with 4000 images in each folder. That resulted in 100\*4000\*3 = 1.2M images.
- This made it easy to access the data, in an ordered way.
- Time Management on Colab The next big issue was time. Colab allows only 8 hrs of continuous usage. Initially when I tried to run the model with 3000 images, that itself was taking around 40 min. As the no. of epochs and dataset were increasing, it was taking close to 8 hrs per run. Creating multiple accounts and parallel processing was a easy to go way to address this issue.
- Another way would be to save the model and run from the place left. This also addresses Colab Crash which happened very often at the end.

Time consumption: 45 min – (12\*3) 36k images.

The link to the modules is:
  **Modules:**
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/UnetModel.py
  
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Train_model.py
  
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Train_model.py
  
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Test.py
  
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Train.py
  
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Results.py
  
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Data_Transform.py
  
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Custom_Dataset.py

  
The link to the modularised code is:
  **Code Implementation:**
   https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Code_Implementation/S15_FinalModular.ipynb
  
  **Results:**
  ![Actual and Predicted Mask](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Images/Result_mask.png)
  
  ![Actual and Predicted Depth](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Images/Result_Depth.png)
  
  
  
  
  
  
  
  
  
  
  
  
  
      
