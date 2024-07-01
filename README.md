# Low-Light-Image-Enhancement-Using-GAN

Then main obective of this project is to solve the challenge of enhancing low-light images using Generative Adversarial Networks (GANs). The study primarily focuses on improving the performance of GAN-based low light image enhancement models through hyperparameter tuning and advanced image preprocessing techniques.

Dataset : 

The work in this research is conducted using the publicly accessible LOL dataset, which consists of 500 image pairs with low and normal light divided into 485 training pairs and 15 testing pairs.Every image in the dataset has a resolution of 400 x 600.

Dataset Link: https://paperswithcode.com/dataset/lol

Below is a sample image from the dataset and its corresponding image that is captured in a well lit environment :
![Dataset:Sample Input Image vs GroundTruth](https://github.com/pratikpandey13/Low-Light-Image-Enhancement-Using-GAN/blob/main/Images/Sample%20Image%20vs%20GroundTruth%20Dataset.jpeg)

## Model Architecture

![GAN Architecture](https://github.com/pratikpandey13/Low-Light-Image-Enhancement-Using-GAN/blob/main/Images/GAN_Architecture.jpeg)

-- Generator Architecture
The Pix2PixHD generator model is designed for high-resolution image synthesis and semantic 
manipulation using conditional Generative Adversarial Networks (cGANs). It significantly 
improves upon the original Pix2Pix framework by introducing a more sophisticated 
architecture capable of handling high-resolution images. 

Below is the genrator architecture diagram:
![Generator Architecutre](https://github.com/pratikpandey13/Low-Light-Image-Enhancement-Using-GAN/blob/main/Images/Generator%20Architecture.jpeg)


![Discriminator Architecture](https://github.com/pratikpandey13/Low-Light-Image-Enhancement-Using-GAN/blob/main/Images/Discrimator%20Architecture.jpeg)

## Conclusion

The study on low-light image enhancement using the Pix2PixHD model with the LOL dataset highlights significant advancements in image processing. By training on diverse, high-quality paired images, the model effectively enhances low-light images, maintaining color fidelity and detail, and shows promise for future improvements in computational photography.

![Output of the Project:Input Image vs Ground Truth vs Image Generated by the Model](https://github.com/pratikpandey13/Low-Light-Image-Enhancement-Using-GAN/blob/main/Images/Input%20Image%20vs%20Ground%20Truth%20vs%20Output%20%20.jpeg)




