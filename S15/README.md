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
      
  # S15
  **Modules:**
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/UnetModel.py
  
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Train_model.py
  
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Train_model.py
  
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Test.py
  
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Train.py
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Results.py
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Data_Transform.py
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Modules/Custom_Dataset.py
  
  **Code Implementation:**
  https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/oS15model_4kE1_1_F1.ipynb
  
  **Results:**
  [].![https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Images/Result_mask.png]
  
  ![https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S15/Images/Result_Depth.png]
  
  
  
  
  
  
  
  
  
  
  
  
  
      