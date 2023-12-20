import cv2
import insightface
from insightface.app import FaceAnalysis

# Creating an instance of FaceAnalysis class
app = FaceAnalysis('buffalo_l')

# Initializing the instance
app.prepare(ctx_id=-1, det_size=(640, 640))

# Loading the image and detecting the face
image = cv2.imread('/Users/harsh/DataSpellProjects/Swap With Elon/elon-musk.jpg')

# Getting all the faces from the image
image_faces = app.get(image)

# Getting the face of Elon Musk
elon_musk_face = image_faces[0]

# Creating model instance for swapping faces
swapper = insightface.model_zoo.get_model(
    '/Users/harsh/DataSpellProjects/Swap With Elon/inswapper_128.onnx',
    providers=['CoreMLExecutionProvider']
)

# Capturing the video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Reading the frame
    status, frame = cap.read()

    # Breaking the loop if the frame is not captured
    if not status:
        break

    # Getting all the faces from the frame
    frame_faces = app.get(frame)

    # Checking if any face is detected
    if len(frame_faces) > 0:
        # Swapping the first face from the frame with Elon Musk's face
        frame = swapper.get(frame, frame_faces[0], elon_musk_face, paste_back=True)

    # Displaying the frame
    cv2.imshow("Frame", frame)

    # Pressing ESC key to exit
    if cv2.waitKey(1) == 27:
        break

# Releasing the video capture
cap.release()

# Destroying all windows
cv2.destroyAllWindows()
