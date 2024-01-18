
#  Face Recognition and Verification with FaceNet model

This repository provides demonstration of how to build accurate face recognition system using the state-of-the-art FaceNet model. The system is deployed in Kivy and do following:
- Face detection: detect and localize faces in the given image using MTCNN 
- Face verification: task that involves verifying the identity of a person with a given image. Is this the same person?
- Face recognition: task that involves recognizing the identity of a person from a database of known faces. Who is this person?
- One-shot-Learning: Machine learning approach where a model is trained to recognize or classify objects based on a single or very few examples per class  


## Acknowledgements

Usefull resources
* [Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering][1]
* [Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). DeepFace: Closing the gap to human-level performance in face verification][2]
* [Gregory Koch, Richard Zemel, Ruslan Salakhutdinov (2015). Siamese Neural Networks for One-shot Image Recognition][3]
* [Jason Brownlee (2020). How to Develop a Face Recognition System Using FaceNet in Keras][4]

Git hub repositories 
* [TensorFlow to Keras][5]
* [Official FaceNet Repository][6]

[1]: https://arxiv.org/pdf/1503.03832.pdf
[2]: https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf
[3]: https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
[4]: https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
[5]: https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb
[6]: https://github.com/davidsandberg/facenet
## Demo
![fast](https://private-user-images.githubusercontent.com/62949953/297286942-b7d6a8e1-7300-4d10-b000-f240e0d5d995.gif?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDU1ODQ1NDksIm5iZiI6MTcwNTU4NDI0OSwicGF0aCI6Ii82Mjk0OTk1My8yOTcyODY5NDItYjdkNmE4ZTEtNzMwMC00ZDEwLWIwMDAtZjI0MGUwZDVkOTk1LmdpZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAxMTglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMTE4VDEzMjQwOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWJkOTFiNDk5Y2QxZTA3YWU3Mzg0OWVlMWI5OWE3YjIxZmY2YzQ3NjQ0Y2Q4NTc5NDMxODI1NTllYTFiZTZkMjgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.NvpAjFJKxM-kNne9B6modtB_7ZbweyPRM3QFM2VxkbM)


## Run Locally

Clone the project

```bash
  git clone https://github.com/NurNazaR/Face-Recognition-and-Verification-with-FaceNet-model.git
```

Go to the project directory

```bash
  cd face_id
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Upload your photo to the  'database (people)' folder and  Rename the file 'fullname.jpg'

Run the face recognition application

```bash
  python app/face_recognition.py
```

Modify the variable self.identity in the face_verification.py file to your full name

Run the face verification application
```bash
  python app/face_verification.py
```


## License

[MIT](https://choosealicense.com/licenses/mit/)

