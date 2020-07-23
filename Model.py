""" Script to perform face recognition and ID

Inspired by: https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
facenet weights: https://drive.google.com/open?id=1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn



"""

from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import cv2
import numpy as np
import pickle
import os
from sklearn.svm import SVC


class FaceID:
    svm_path = './svm.pickle'
    face_data_path = './face_data.npz'
    facenet_path = './facenet_keras.h5'

    def __init__(self):
        # localizer
        self.mtcnn = MTCNN()

        # facenet model
        self.facenet = load_model(self.facenet_path)

        # load or create an SVM classifier
        if not os.path.exists(self.svm_path):
            self.classifier = SVC()
        else:
            self.classifier = self.load_pickle(self.svm_path)

        self.face_data = {}

    def get_faces(self, image):
        """ will return a numpy array of faces for facenet input ie: (faces, 160, 160, 3) """

        # get results from facial localizer
        results = self.mtcnn.detect_faces(image)

        # get centroid locations
        centroids = self.get_centroids(results)

        # clip into square images
        return np.array([self.clip_face(image, centroid) for centroid in centroids]), results

    def add_to_dataset(self, image):
        """ add data for svm training """
        faces, results = self.get_faces(image)
        x, y = np.array([]), np.array([])

        # load saved face data if there is any
        if os.path.exists(self.face_data_path):
            # trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
            self.face_data = np.load(self.face_data_path)

        # get old data, if not available
        if 'arr_0' in self.face_data.keys() and 'arr_1' in self.face_data.keys():
            x, y = self.face_data['arr_0'], self.face_data['arr_1']

        # show the image and ask for a name
        for face, result in zip(faces, results):
            # draw a rectangle around the image to highlight it
            x1, y1, x2, y2 = self.parse_result(result)
            image_query = np.copy(image)
            image_query = cv2.rectangle(image_query, (x1, y1), (x2, y2), (255, 255, 255), 3, )
            image_query = cv2.putText(image_query, 'Who is this?', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

            # show the image to the user
            cv2.imshow('Who is this?', image_query)
            cv2.waitKey(1000)

            name = input('Who\'s face is this? ')

            # ask the user for the name and ad that to the data
            y = np.append(y, name)

            # clear window after we have our answer
            cv2.destroyAllWindows()

            # add the facenet embedding to the training data
            embedding = self.facenet(np.array([face])).numpy()
            x = np.append(x, embedding)

        # save the new training data as a compressed .npz file
        np.savez(self.face_data_path, x, y)

    def get_centroids(self, results):
        """ extracts centroids from mtcnn results """
        centroids = []

        for result in results:
            x1, y1, x2, y2 = self.parse_result(result)
            centroids.append((x2 - x1, y2 - y1))

        return centroids

    def detect_faces(self, fp):
        """ display results of facial ID


        """
        image = cv2.imread(fp)
        # saved_embeddings = self.load_pickle(self.embedding_save_path)
        faces, results = self.get_faces(image)

        for face, result in zip(faces, results):
            x1, y1, x2, y2 = self.parse_result(result)

            # draw a rectangle around the face
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 3, )
            embedding = self.facenet.predict(np.array([face]))[0]

            # get the name for that face and add it to the image
            name = self.classifier.predict([embedding])[0]
            cv2.putText(image, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

        cv2.imshow('', image)
        cv2.waitKey(1000)

    def train_from_file(self, fp):
        image_files = [file for file in os.listdir(fp) if '.jpg' in file]

        for image_file in image_files:
            path = os.path.join(fp, image_file)
            self.add_to_dataset(cv2.imread(path))

        # retrain linear svc
        self.face_data = np.load(self.face_data_path)
        x, y = self.face_data['arr_0'], self.face_data['arr_1']
        x = x.reshape(x.shape[0] // 128, 128)
        self.classifier.fit(x, y)
        self.save_pickle(self.classifier, self.svm_path)

    @staticmethod
    def clip_face(image, centroid):
        """ returns a square image of a face given its center position. Edge cases will be scaled to be square,
        so model results are expected to be worse around the edges of the frame.

         It will also normalize the pixels
         """

        # get min max coordinates
        cx, cy = centroid
        x1 = cx - 80
        x2 = cx + 80
        y1 = cy - 80
        y2 = cy + 80

        # handle edge-cases
        if x1 < 0:
            x1 = 0
        elif x2 > image.shape[0]:
            x2 = image.shape[0]
        if y1 < 0:
            y1 = 0
        elif y2 > image.shape[1]:
            y2 = image.shape[1]

        # clip out the face
        face = image[x1:x2, y1:y2]

        # check to see if the image is indeed square, resize if not
        if image.shape[1] != 160 and image.shape[2] != 160:
            face = cv2.resize(face, (160, 160))

        # normalize face pixels for model input
        mean, std = face.mean(), face.std()
        face = (face - mean) / std

        return face

    @staticmethod
    def parse_result(result):
        """ utility function to return min x, min y, max x, max y from mtcnn results """

        x1, y1, width, height = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        return x1, y1, x2, y2

    @staticmethod
    def save_pickle(obj, filename):
        """ Simple utility function to save a pickle file

        Args:
            obj: (obj): almost any python object
            filename (str): path where you would like to save the .pickle file. Extension .pickle must be there
        """
        with open(filename, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle(filename):
        """ Simple utility function to load a pickle file

        Args:
            filename (str): path to the .pickle file in question
        """
        with open(filename, 'rb') as handle:
            return pickle.load(handle)
