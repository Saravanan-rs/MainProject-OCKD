import cv2
import dlib
import numpy as np

class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # Store the desired output left eye position and face width + height
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # If the desired face height is None, set it to be the desired face width
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, pos):
        leftEyeCenter = pos[0]
        rightEyeCenter = pos[1]

        # Compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # Compute the desired right eye x-coordinate based on the desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # Determine the scale of the new resulting image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0]) * self.desiredFaceWidth
        scale = desiredDist / dist

        # Compute center (x, y)-coordinates between the two eyes in the input image
        eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
                      int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))

        # Grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # Update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # Apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        # Return the aligned face
        return output

def shape_to_np(shape, dtype="int"):
    # Facial landmark coordinates: shape -> numpy array
    coords = np.zeros((5, 2), dtype=dtype)
    for i in range(0, 5):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def main():
    aligner = FaceAligner()

    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        rects = detector(gray, 0)

        # Iterate over detected faces
        for rect in rects:
            # Predict facial landmarks
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            pos = np.array([0.5 * (shape[1] + shape[0]), 0.5 * (shape[3] + shape[2])]).astype(int)

            # Align and crop the face
            aligned_face = aligner.align(image=frame, pos=pos)

            # Save the aligned face
            cv2.imwrite('aligned_face.jpg', aligned_face)

            # Display the cropped face
            cv2.imshow('Cropped Face', aligned_face)

        # Display the original frame
        cv2.imshow('Original Frame', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
