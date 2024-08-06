# Image Classification Model Comparison with PyTorch

This repository contains a Jupyter Notebook that implements an image classification model using the ResNet18 architecture, as described in the research paper by He, Kaiming, et al. “Deep Residual Learning for Image Recognition.” arXiv, 10 Dec. 2015, arxiv.org/abs/1512.03385. 

<img src='https://miro.medium.com/v2/resize:fit:720/format:webp/1*rrlou8xyh7DeWdtHZk_m4Q.png' width='800'>

In addition, this notebook contains another image classification model using the Vision Transformer (ViT) architecture, as described in the research paper by Dosovitskiy, Alexey, et al. “An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale.” arXiv, 22 Oct. 2020, https://arxiv.org/abs/2010.11929. Accessed 5 July 2024. 

<img src='https://miro.medium.com/v2/resize:fit:720/format:webp/1*SoXHGxDPUqFQHFbJKYoVLg.png' width='800'>

## Project Overview

The goal of this project is to build two models that can identify an image and what class it belongs to. I am comparing the performance between ResNet and Vision Transformers (ViT) architectures for image classification.

## Training and Evaluation

- **Training:**
   - Both models are trained using the SGD optimizer with an initial learning rate of 0.005.
   - Hyperparameters:
     - **Epochs**: 11
     - **Batch Size**: 10
     - **Number of Classes**: 6
     - **Embed Dimensions**: 512(ResNet), 768(ViT)
   - Masked cross-entropy Loss is used as the loss function we minimize.
   - We use a masked accuracy function to evaluate how well the model generates the highest probability tokens during training.

## Results on Unseen Images

-**ResNet**:
Accuracy of the networks: 89.78%

-**ViT**:
Accuracy of the networks: 91.96%

In conclusion, I recommend using ResNet models when training a machine to classify images. I think this because while it is known that ResNet models train faster and Vision Transformer models are known for their strong performance in certain scenarios, it does not offer a significant improvement in accuracy in this instance. Therefore, the quicker training time of ResNet makes it the more efficient choice without sacrificing substantial model performance.
  
## Getting Started
1. **Download the Garbage-Classification-Image Dataset:**
   - Obtain the dataset from [Kaggle](https://www.kaggle.com/datasets/saumyamohandas/garbage-classification-image-dataset) and place it in the project directory.

2. **Set Up Google Colab (Optional):**
   - If using Google Colab, mount your Google Drive and store your Kaggle credentials using `google.colab.userdata`.

3. **Run the Jupyter Notebook:**
   - Open and execute the `Waste Image Classification` notebook to train and evaluate the model.
  
## Acknowledgments

- He, Kaiming, et al. “Deep Residual Learning for Image Recognition.” arXiv, 10 Dec. 2015, arxiv.org/abs/1512.03385.
- Dosovitskiy, Alexey, et al. “An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale.” arXiv, 22 Oct. 2020, https://arxiv.org/abs/2010.11929. Accessed 5 July 2024. 
- The Garbage-Classification-Image dataset is used for training and evaluation.

