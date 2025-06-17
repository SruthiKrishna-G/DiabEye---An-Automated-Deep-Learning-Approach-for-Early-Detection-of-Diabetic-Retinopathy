# DiabEye---An-Automated-Deep-Learning-Approach-for-Early-Detection-of-Diabetic-Retinopathy

## Project Overview
DiabEye is a deep learning-based system designed for early detection and classification of diabetic retinopathy (DR) from retinal fundus images. Using the DenseNet121 architecture, the model helps healthcare professionals by providing fast and accurate diagnostic predictions across five DR severity levels.

## Features
- Automated classification of diabetic retinopathy severity: No DR, Mild, Moderate, Severe, and Proliferative DR.
- User-friendly web application for image upload and real-time prediction.
- Ensemble learning techniques to improve prediction accuracy.
- Visual output graphs displaying prediction probabilities and classification results.

## Dataset
This project uses the publicly available *DDR* dataset, which consists of a large collection of multi-source dermatoscopic images.  
You can download the full dataset from [Kaggle](https://www.kaggle.com/datasets/mariaherrerot/ddrdataset).  

![image](https://github.com/user-attachments/assets/c0777158-d63d-43df-81b8-cc6fe1217da5)

Note: Only sample images are included in this repository due to dataset size constraints.


DiabEye/

──> Sample data/              # Sample dataset images and labels

──> requirements.txt          # Python dependencies

──> Diabeye_Workflow.jpg      # Structured pipeline

──> models/                   # Trained model files and training scripts

──> diabeye_main.py           # Main execution script

──> app/                      # Web application source code

──> README.md                 # Project documentation
