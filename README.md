# Keras FaceID
This repo contains a very lightweight implementation of a facial recognition pipeline. It is designed to persist, and learn from new data
as it is added. I wanted this to be integrated with another project, so I have not added an argparse. 

Here is what the output looks like:
 ![some majestic pretend-infantry repulsing cavalry](./square.jpg)


## How to Use 
 1. Install requirements from requirements.txt
 2. Grab the facenet_keras.h5 model from https://drive.google.com/open?id=1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn
 3. Instantiate an instance of the FaceID class
 4. Point the train_from_file method to a folder with images containing faces
 5. Type in the subject's name when prompted 
 6. Any instance of FaceID will use the trained model, allow one to detect faces from a frame using the detect_faces function these files will be created and loaded automagically 
 
 
## Future Feature Wishlish
- PCA + k-means clustering function to classify names from folder to create an unsupervised dataset
- Video capability 
- Bounding box tracking on video to quickly create a dataset for new subject
- Data augmentation/balancing before training  

 
## Reference
 I was heavily inspired by the pipeline described on machinelearningmastery.com, I felt it was an efficient solution that would be as minimal as possible
 
 
https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
