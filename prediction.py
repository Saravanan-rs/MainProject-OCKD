import torch
import torchvision.transforms as transforms
from PIL import Image
from src.arch import TeacherNet
from architecture import StudentNet
import torch.nn.functional as F

# Define a transformation to preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((256,256)),  # Resize image to the input size of the model
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(            # Normalize image data
        mean=[0.485, 0.456, 0.406],  # These values are used for the ImageNet dataset
        std=[0.229, 0.224, 0.225]
    ),
])

# Load the image
image_path = 'aligned_face.jpg'
image = Image.open(image_path).convert('RGB')

# Preprocess the image
input_image = preprocess(image).unsqueeze(0)  # Add a batch dimension

# Load the teacher and student models
teacher = TeacherNet(ic=3)
student = StudentNet(ic=3)
teacher.load_state_dict(torch.load(r'models\train_teacher\casia\casia_0\seed_1\19\teacher.pt', map_location=torch.device('cpu')))
student.load_state_dict(torch.load(r'models\train_ideal_student_10\casia\client_1\seed_1\0\student.pt', map_location=torch.device('cpu')))

# Set the models to evaluation mode
teacher.eval()
student.eval()

# Forward pass through the teacher network
with torch.no_grad():
    teacher_output, _, _, _ = teacher(input_image)

# Forward pass through the student network
with torch.no_grad():
    student_output, _, _, _ = student(input_image)

# Compute the similarity score between the teacher and student predictions
def compute_similarity_score(preds1, preds2):
    preds1 = preds1.view(preds1.shape[0], -1)
    preds2 = preds2.view(preds2.shape[0], -1)
    similarity = 1 - F.cosine_similarity(preds1, preds2, dim=1)
    return similarity

similarity_score = compute_similarity_score(teacher_output, student_output)
print("Similarity Score:", similarity_score.item())
