import numpy as np
from tensorflow.keras.datasets import mnist # type: ignore # Using Keras just to download easily
# matplotlib for visualization
import matplotlib.pyplot as plt
import random
import cv2


def openCamera(factor):

    # Open the default camera (0 = built-in webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # get height and width
        h, w = gray.shape
        # define the size of the square
        size = min(h, w)//factor
        y1 = h//2-size//factor
        x1 = w//2-size//factor
        # crop the image to a square
        roi = gray[y1:y1+size, x1:x1+size]

        # resize the image
        roi = cv2.resize(roi, (28,28))
        # invert image
        roi = cv2.bitwise_not(roi)
        roi_min = np.min(roi)
        roi_max = np.max(roi)
        
        roi = (roi - roi_min)/(roi_max - roi_min)
        
        display = roi

        # reshaping for input 
        final = roi.reshape(-1, 784)

        output = feedForward(final)
        prediction = np.argmax(output[4])
        confidence = np.max(output[4])

        threshold = 0.58
        if(confidence<=threshold):
            res = "No number detected"
        else:
            res = f"Prediction: {prediction} Confidence:{confidence}"
        
        cv2.rectangle(frame, (x1, y1), (x1+size, y1+size), (0,255,0), 2)
        cv2.putText(frame, res, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 5)

        cv2.imshow("MNIST Camera", frame)
        cv2.imshow("input for nn", display)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()






def sigmoid(z):
    z = np.clip(z, -500, 500) 
    return 1 / (1 + np.exp(-z))

def feedForward(data):
    a1=data.T

    z2=weights1@a1 + bias1
    a2=sigmoid(z2)

    z3=weights2@a2 + bias2
    a3=sigmoid(z3)

    z4=weights3@a3 + bias3
    a4=sigmoid(z4)

    z5=weights4@a4 + bias4
    a5=sigmoid(z5)

    return a1,a2,a3,a4,a5

# load the neural network saved in brain.npz
def load_model(filename="brain.npz"):
    global weights1, weights2, weights3, weights4
    global bias1, bias2, bias3, bias4
    
    # Load the archive
    data = np.load(filename)

    # Extract variables by the names we gave them
    weights1 = data['w1']
    weights2 = data['w2']
    weights3 = data['w3']
    weights4 = data['w4']
    
    bias1 = data['b1']
    bias2 = data['b2']
    bias3 = data['b3']
    bias4 = data['b4']
    
    print(f"Brain loaded from {filename}")

# function to calculate the accuracy
def accuracy(data,label):
    a,b,c,d,out = feedForward(data)
    predictions = np.argmax(out, axis=0)
    correct_predictions = np.sum(label == predictions)
    percent = correct_predictions/len(predictions)*100
    return percent

# function to calculate the confusion matrix
def confusion(data,label):
    confused = np.zeros((10,10),dtype=int)
    a,b,c,d,out = feedForward(data)
    predictions = np.argmax(out, axis=0)
    for i in range(len(label)):
        y = label[i]
        x = predictions[i]
        confused[y][x] += 1
    return confused

# Helper function to plot the confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap='Blues') # 'Hot' or 'Greens' also look good
    
    # Add numbers to the squares
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha='center', va='center', 
                     color='white' if cm[i, j] > 500 else 'black')

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.colorbar()
    plt.show()

# function to get a random image from the test dataset and predict digit for it
def testrandomimage(data,l):
    idx = random.randint(0,len(data)-1)
    number = data[idx:idx+1,:]
    out = feedForward(number)
    prediction = np.argmax(out[4],axis=0)
    
    current_image = number.reshape(28, 28)
    plt.imshow(current_image, cmap='gray')
    plt.title(f"AI: {prediction} | Actual {l[idx]}")
    plt.show()
    


# This downloads the files from the internet and puts them into 4 NumPy arrays
(x_train_raw, y_train), (x_test_raw, y_test) = mnist.load_data()
x_train_flat = x_train_raw.reshape(-1, 784)
x_test_flat = x_test_raw.reshape(-1, 784)
x_test_flat=x_test_flat.astype('float32') / 255.0 #→ Best practice. Creates Float32 (Efficient memory, GPU-ready).
x_train_flat=x_train_flat.astype('float32') / 255.0
#we successfully loaded the data

# Load the neural network saved in "brain.npz"
load_model()

while(True):
    print(" Press 1 to calculate accuracy \n Press 2 to get a random image and predict its digit \n Press 3 to get the confusion matrix \n Press 4 to open camera")
    choose = int(input())
    if(choose==1):
        print(accuracy(x_test_flat, y_test))
    elif(choose==2):
        testrandomimage(x_test_flat, y_test)
    elif(choose==3):
        plot_confusion_matrix(confusion(x_test_flat, y_test))
    elif(choose==4):
        print("Enter crop factor")
        inp = int(input())
        openCamera(inp)
    else:
        print("Invalid choice \n Try again")
            
