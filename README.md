# Neuro-Diagnostic

Home Page            |  Scan Result
:-------------------------:|:-------------------------:
![HomePage](assets/HomePage.jpg)  |  ![ScanResult](assets/ScanResult.jpg)

## Inspiration ❗
Alzheimer's disease is the most common type of dementia, and can lead to the loss of the ability to carry on a conversation and respond to the environment. Neuroscientists can spend hours studying scans and various biomarkers to accurately diagnose patients with early stages of Alzheimer's. However, with the rapid advancement of machine learning software I reasoned that there must be more efficient ways to diagnose a patient with Alheimer's using imaging technolgies like MRI.

## What it does 💭
Neuro-Diagnostic accepts a patients MRI scan and runs it through a multi-classification machine learning model to output whether the patiens is cognitively normal, mildly cognatively impaired, or has Alzheimer's. 

## How I built it ❓
This app was built using only Python for the computational logic, with a Flask backend complemented by a simple React frontend. Once a user uploads an MRI scan in the form of a 3D NIFTI file, the image gets passed through the preprocessing pipeline. In the pipeline the image is resampled, registered to a common atlas, whitened, and finally converted to a 2D image by taking multiple slices at different regions of the brain. To aid in preproccessing various libraries and open source tools were utilized, including SimpleITK and The Deep Learning Toolkit for Medical Imaging(DLTK). The processed 2D images are then written to TFRecords and made into a dataset upon which a CNN is trained. 

## How well does it work ❓
The trained model has an accuracy of about 67%, however using accuracy as a metric in this situation is not the best, due to a significant class imbalance. There are far more MCI sampled than those with AD. A better metric, and one which conveys a more successful result is the ROC curve. Although still not anywhere near the level required for this to be used in a workplace setting, the ROC curves for each class lie well above that which would be achieved by a random classifier. 

## Challenges 😣
One of the most challenging aspects of this project was figuring out how to perform preprocessing on 3D MRI scans. A lot of time was spent looking at the scans with different visualization software and learning about the file format. The biggest challenge after this was trying out various architectures of the CNN, although the problem here largely stemmed from computational limitations. 

## What I learned 🧠
I learned many different things by working on this project. Some of these included preprocessing techniques for images as well as how to build a Tensorflow input pipeline with TFrecords. 

## What's next for Neuro-Diagnostic ▶️
The next planned step is to implement the ability to diagnose brain cancer. 


