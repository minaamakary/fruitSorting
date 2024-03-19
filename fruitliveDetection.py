import cv2
import numpy as np
from keras.models import model_from_json

fruit_dictionary = {
    0: 'Apple',
    1: 'Banana',
    2: 'Orange'
}

def crop_image(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use a simple threshold to segment the cards from the background
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to find bounding boxes around cards
    image_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        image_boxes.append((x, y, x+w, y+h))

    # Return the bounding boxes of detected cards
    return image_boxes



json_file = open('fruit_CNN_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("fruit_CNN_Model_complete.h5")
print("Loaded model from disk")

# pass here your video path
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    # Crop images from the frame
    image_boxes = crop_image(frame)
    # Perform any required preprocessing on the frame before prediction
    for box in image_boxes:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Fruit Detection", frame)

    # Assuming your model expects a certain input shape, resize and normalize the image
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = resized_frame / 255.0  # Normalize pixel values to be between 0 and 1
    input_data = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = emotion_model.predict(input_data)
    predicted_class = np.argmax(prediction)
    
    # Get the corresponding card label from the dictionary
    fruit_label = fruit_dictionary[predicted_class]
    
    # Display the card label on the frame
    cv2.putText(frame, str(fruit_label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Fruit Prediction", frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

