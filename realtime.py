import cv2
import torch
import numpy as np
from architecture import TeacherNet  # Assuming you have the TeacherNet architecture defined in 'architecture.py'
import torchvision.transforms as transforms
from PIL import Image


def preprocess_image(frame):
    # Convert frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize and normalize the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Assuming your model takes input of size 256x256
        transforms.ToTensor(),
    ])
    image = Image.fromarray(frame_rgb)
    image = transform(image)
    image = (image - 0.5) / 0.5  # Assuming the same normalization as in the dataset
    image = image.unsqueeze(0)  # Add batch dimension
    return image


def load_teacher_model(model_path):
    # Load the trained TeacherNet model on CPU
    teacher_model = TeacherNet()
    teacher_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    teacher_model.eval()
    return teacher_model

def predict_from_webcam(model, threshold):
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Preprocess the frame
        image = preprocess_image(frame)

        # Predict using the teacher model
        with torch.no_grad():
            teacher_output, _, _, _ = model(image)  # Assuming your model returns multiple outputs

        # Take the mean of the first output tensor
        teacher_output_mean = torch.mean(teacher_output[0])

        # Classify as real or spoof based on the threshold
        if teacher_output_mean.item() < threshold:
            prediction = "Real"
        else:
            prediction = "Spoof"

        # Display the prediction on the frame
        cv2.putText(frame, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Webcam', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Path to the trained teacher model
    model_path = r'models\train_teacher\casia\casia_0\seed_1\20\teacher.pt'

    # Load the teacher model
    teacher_model = load_teacher_model(model_path)

    # Define the threshold for classification
    threshold = 0.3453

    # Predict from webcam
    predict_from_webcam(teacher_model, threshold)

if __name__ == "__main__":
    main()
