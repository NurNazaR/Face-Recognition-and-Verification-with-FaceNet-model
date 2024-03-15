
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
## Demo (click to view video)
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/eXjEtZkBgeg/0.jpg)](https://www.youtube.com/watch?v=eXjEtZkBgeg)



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

