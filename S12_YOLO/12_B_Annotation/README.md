**Assignment B**

1.Download 50 images of dogs.

2.Use [this](http://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) to annotate bounding boxes around the dogs.

3.Download JSON file.

4.Describe the contents of this JSON file in FULL details.

5.Refer to this [tutorial](https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203). Find out the best total numbers of clusters. 

6.Upload link to your Colab File uploaded to GitHub.


[Dataset](https://github.com/JahnaviRamagiri/EVA-B2/tree/master/S12_YOLO/12_B_Annotation/Dataset)

![](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S12_YOLO/12_B_Annotation/Dataset/dog_pics.png)

Attributes:

![](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S12_YOLO/12_B_Annotation/Dataset/Bounding%20box.png)


**JSON File Description**

{"D01.jpg7176":{"filename":"D01.jpg","size":7176,"regions":[{"shape_attributes":{"name":"rect","x":42,"y":43,"width":160,"height":102},"region_attributes":{"Class":"Dog","ID":"D01","Caption":"Mamma Dog playing with it's Pup!"}},{"shape_attributes":{"name":"rect","x":158,"y":44,"width":114,"height":93},"region_attributes":{"Class":"Dog","ID":"D01","Caption":"Pup!"}}],"file_attributes":{"caption":"","public_domain":"no","image_url":""}}

Filename: D01.jpg

File size: 7176 KB

**Regions:**

**Shape Attributes:** (2 Bounding Boxes are available in this image)

*box1*

Bounding Box type: Rectangular

Coordinates of Top Left corner: (42,43)

Width of bounding box: 114

Height of Bounding Box: 102

*box2*

Bounding Box type: Rectangular

Coordinates of Top Left corner: (158,44)

Width of bounding box: 114

Height of Bounding Box: 93

**Region Attributes:**

Class: Dog

ID: (named after the dog image number) : D01

Caption:"Mamma Dog playing with it's Pup!"

**File Attributes: **

Public domain: No

Image_url: Not provided

[Annotations.json](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S12_YOLO/12_B_Annotation/Dog_annotation_json.json)

![](https://github.com/JahnaviRamagiri/EVA-B2/blob/master/S12_YOLO/12_B_Annotation/Dataset/dog_annt.png)