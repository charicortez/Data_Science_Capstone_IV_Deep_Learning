# Implementing CNNs, ViT, Transfer Learning, and Fine Tuning in the detection of human skin cancer

## Background: AI, CNNS, and breast cancer patients
Convolutional Neural Networks (CNNs) are the go-to for classification in image analysis. AI has markedy augmented the care of breast cancer patients:
* **20% more cancers detected** than the Radiologist-only group
* **2.6%% improvement in the detection of breast cancer** than a Radiologist alone
* **~6% lower rate of false positives** in the US and 1.2% in the UK


## Project Intro: CNNs, ViTs, Computer-Aided Diagnosis (CAD) and skin cancer patients
<img width="214" alt="Screenshot 2024-11-14 at 11 31 15 AM" src="https://github.com/user-attachments/assets/7b836bbe-d550-4087-b022-6fa067769744">

Globally, skin cancer is highly signficant and widespread with millions of cases worldwide and consistently rising incidence in recent years. Presently, diagnosis depends on a Dermatologists' professional knowledge and visual examination, which is inherently subjective and prone to diagnostic inaccuracies. Skin tumors, whether malignant or benign, have a highly hetrogenous and diverse presentation (as seen above). As such, there is an *urgent medical need* for **more accurate and automated mechanisms for identifying skin cancer**. CAD frameworks have addressed this need, providing mehotds that avoid subjectivity and changes in precision that could occur between Dermatologists. 

CNNs and Vision Transformers (ViTs) have shown remarkable promise for *improving the accuracy of skin cancer recognition*. The combination of *deep learning and optimization methodologies,* such as Transfer Learning and Fine-Tuning, represent an active, robust area of research in the efforts to develop enhanced techniques in skin cancer diagnosis. 

## Combining deep learning and optimization methodologies to enhance CAD of skin cancer
<img width="1476" alt="Screenshot 2024-11-14 at 11 36 44 AM" src="https://github.com/user-attachments/assets/01495308-749f-44b6-a5de-2eae4b269ce3">

## Project Data
Processed pictures of moles from the [International Skin Imaging Collaboration (ISIC) Archive](https://www.isic-archive.com/). 
<img width="518" alt="Screenshot 2024-11-14 at 11 41 16 AM" src="https://github.com/user-attachments/assets/25b87b05-14df-4040-8b99-ddf487cfd8aa">
<img width="308" alt="image" src="https://github.com/user-attachments/assets/221dbbff-eb30-4f64-8b1f-cca79dadeddb">

To prepare the data for Training, the Train data was split into New Train and Validation. These data were class-balanced and used to create New Train and Validation dataloaders compatible with the PyTorch pipeline. 

<img width="283" alt="image" src="https://github.com/user-attachments/assets/cef929c8-50ab-48eb-ac2d-975071624a45">

*To help make the model more robust/invariant, and to help minimize overfitting of the model to the Train data*, PyTorch transformations/augmentations were applied to the images in the New Train dataset. Random Vertical Flip and Random Horizontal Flip were applied to only New Train and not Validation; industry best practice requires the application of an "empty" tranformation/augmentation function if there are more than two classes in this dataset (only two exist in this task). 

## CNN Model: Establishing baseline performance metrics
What's my base? Establishing a "base" model and baseline performance metrics when developing and evaluating models is industry best practice. Establishing a base and executing A/B testing when developing and evaluating models allows direct comparison of performance metrics across models and quantification of differences between model iterations. 

### The CNN model is converging and learning well

<img width="596" alt="image" src="https://github.com/user-attachments/assets/8e23804d-4c04-4db9-8ace-397a3522942f">

### Baseline performance metrics show model is performing well

<img width="316" alt="image" src="https://github.com/user-attachments/assets/c7f7ee50-ba0b-48a9-81b7-003b9acdc35b">
<img width="278" alt="image" src="https://github.com/user-attachments/assets/8b362123-7f04-4d7a-94d0-fd3f9636cd6a">

### CNN ROC curves are in the ideal spot

<img width="434" alt="image" src="https://github.com/user-attachments/assets/4c37f6d7-953e-464a-a620-fb56d62dd7d8">

## ResNet model: Transfer Learning; from Hugging Face Computer Vision course Transformers library
The ResNet model used the *same Loss function, Optimizer, Learning Rate, Learning Rate Scheduler, and Number of epochs* that were used in the CNN model.

### Validation data shows model is learning and performing well

<img width="596" alt="image" src="https://github.com/user-attachments/assets/d70ba6ca-a1e7-49f0-887b-56ceac627099">

### ResNet has *excellent* performance metrics and less False Negatives/False Malignants

<img width="844" alt="Screenshot 2024-11-14 at 11 55 23 AM" src="https://github.com/user-attachments/assets/b045ae69-d400-4f6a-8ec2-e04968bae2a5">
<img width="675" alt="Screenshot 2024-11-14 at 11 56 00 AM" src="https://github.com/user-attachments/assets/f48bbb98-1891-4506-99b8-fac2c8caeb83">

### ResNet ROC curves are in the ideal spot with minimal overfitting

<img width="394" alt="image" src="https://github.com/user-attachments/assets/b33dfa30-07af-4f95-9ae4-e8565d9a3241">

## ViT Model: Transfer Learning; from Weights & Biases (W&B) AI developer platform
The ViT model used the *same Learning Rate Scheduler, and Number of epochs* that were used in both the CNN and ResNet model.

### ViT is converging and learning, but has lowest accuracy

<img width="414" alt="image" src="https://github.com/user-attachments/assets/6300517a-95ef-47b3-a44b-c6a57aa629cf">

### ViT performance metrics are similar to those of baseline CNN; most FNs

<img width="347" alt="image" src="https://github.com/user-attachments/assets/4c25c835-0e28-499d-8b0f-884f7386de6d">
<img width="691" alt="Screenshot 2024-11-14 at 12 01 22 PM" src="https://github.com/user-attachments/assets/e45e8d72-6327-4de3-a14a-76b9eef43513">

### ViT Validation ROC curve is in the ideal spot

<img width="432" alt="image" src="https://github.com/user-attachments/assets/ccc7bdb4-5b04-4d1a-b3d3-01c12b48e33c">

## Model comparison summary: Selected the best-performing model for Fine Tuning

<img width="1469" alt="Screenshot 2024-11-14 at 12 03 09 PM" src="https://github.com/user-attachments/assets/26b3bc2b-9ea4-41c9-87ed-9bf7421c5c56">

## Fine-Tuning hyperparamters

<img width="1612" alt="Screenshot 2024-11-14 at 12 04 50 PM" src="https://github.com/user-attachments/assets/966577de-7629-4a21-bcca-fd11ba09e367">

Fine-Tuning hyperparameters maintained Accuracy, but *increased False Negatives*; this is not ideal, as False Negative = False Malignant, which is the misclassification of a Malignant tumor as Benign. Misclassifying a Malignant tumor means that skin cancer is not caught or treated.

## ResNet Validation and Testing: Generalization--how does it perform on the Test (i.e., new, unseen) dataset? 
### ResNet generalizes well to new, unseen data
A random sampling of 25 images showed that ResNet **correctly predicted 22 out of 25 cases**

<img width="557" alt="image" src="https://github.com/user-attachments/assets/f8c7798c-d3b0-41a4-930d-207b34d55bd5">

### ResNet also retained its ability to generalize well across both classes in Test data

<img width="472" alt="Screenshot 2024-11-14 at 12 10 04 PM" src="https://github.com/user-attachments/assets/1a0d6e08-45ad-4981-9fb2-142f873e49ac">

## Successfully combined deep learning and optimization methodologies to enhance CAD of skin cancer

<img width="1496" alt="Screenshot 2024-11-14 at 12 11 17 PM" src="https://github.com/user-attachments/assets/bd5241a5-2b54-4132-914f-dae1012a1ce3">



