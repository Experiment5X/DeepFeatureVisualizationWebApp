# DeepFeatureVisualizationWebApp

An application for visualizing a feature map of an image in VGG16. You can edit the parameters used to create the image to change how the feature map is visualized.

##### Learning Rate
The constant by which to multiply the gradient of the difference of the two feature maps with respect to the input image.
##### Image Std Clip
The number of standard deviations from the mean to clip the image pixel values at.
##### Gradient Std Clip
The number of standard deviations from the mean to clip the gradient values at.
##### Number of Epochs
The number of iterations of gradient descent to perform when updating the image.
##### Total Variation Coefficient
The constant by which to multiply the total variation of the image. The total variation is total difference between neighboring pixels.
##### Noise count
The number of noise pixels to add to the image each iteration. The total number of pixels is 150,528 so this number should be of the same magnitude to have a noticeable effect.


### Example Visulizations
![deep feature visualizations](https://i.imgur.com/KlcJbU3.jpg)


### Web App Screenshot
![](https://i.imgur.com/jA9njTH.png)
