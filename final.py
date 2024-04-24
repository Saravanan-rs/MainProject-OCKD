import torch
from PIL import Image
import torchvision.transforms as transforms
from architecture import TeacherNet  # Assuming you have the TeacherNet architecture defined in 'architecture.py'

def load_teacher_model(model_path):
    # Load the trained TeacherNet model on CPU
    teacher_model = TeacherNet()
    teacher_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    teacher_model.eval()
    return teacher_model

def preprocess_image(image_path):
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Assuming your model takes input of size 256x256
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = (image - 0.5) / 0.5  # Assuming the same normalization as in the dataset
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict_single_image(image_path, model, threshold):
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Predict using the teacher model
    with torch.no_grad():
        teacher_output, _, _, _ = model(image)  # Assuming your model returns multiple outputs
        
    # Take the mean of the first output tensor
    teacher_output_mean = torch.mean(teacher_output[0])
    print("Score:", teacher_output_mean.item())
    # Classify as real or spoof based on the threshold
    if teacher_output_mean.item() < threshold:
        return "Real"
    else:
        return "Spoof"

def main():
    # Path to the trained teacher model
    model_path = r'models\train_teacher\casia\casia_0\seed_1\20\teacher.pt'

    # Load the teacher model
    teacher_model = load_teacher_model(model_path)

    # Path to the single image file
    #image_path = r"datasets\casia\rgb\train\7\spoof\2\10.jpg"
    #image_path = r"datasets\casia\rgb\train\15\real\1\5.jpg"
    image_path = "aligned_face.jpg"
    # Predict using the single image
    prediction = predict_single_image(image_path, teacher_model, 0.3453)

    # Print the prediction
    print("Prediction:", prediction)

if __name__ == "__main__":
    main()
