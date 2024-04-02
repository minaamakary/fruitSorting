import cv2
import numpy as np
from keras.models import model_from_json

fruit_dictionary = {
    0: 'Orange',
    1: 'Banana',
    2: 'Apple',
}

















def crop_image(frame):
   # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color of the fruit you want to detect
    lower_color = np.array([20, 100, 100])  # Adjust these values according to the color of the fruit
    upper_color = np.array([30, 255, 255])  # Adjust these values according to the color of the fruit

    # Create a mask using the inRange function
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Perform morphological operations to remove noise and improve the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    image_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        image_boxes.append((x, y, x+w, y+h))

    # Return the bounding boxes of detected fruits
    return image_boxes



json_file = open('fruit_CNN_Model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("fruit_CNN_Model_complete.h5")
print("Loaded model from disk")

###### Testing Part 

# Load a sample test image from your dataset
test_image = cv2.imread('fruitDataset/Banana/banana_53.jpg')  # Provide the path to your test image

# Crop images from the test image
image_boxes = crop_image(test_image)

# Draw rectangles around detected fruits
for box in image_boxes:
    cv2.rectangle(test_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

# Display the test image with detected fruits
cv2.imshow("Test Image with Detected Fruits", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Assuming your model expects a certain input shape, resize and normalize the test image
resized_test_image = cv2.resize(test_image, (224, 224))
normalized_test_image = resized_test_image / 255.0  # Normalize pixel values to be between 0 and 1
input_data = np.expand_dims(normalized_test_image, axis=0)  # Add batch dimension

# Make prediction on the test image
prediction = emotion_model.predict(input_data)
predicted_class = np.argmax(prediction)

# Get the corresponding fruit label from the dictionary
fruit_label = fruit_dictionary[predicted_class]

# Print the predicted fruit label
print("Predicted Fruit:", fruit_label)

###### Testing Part



# pass here your video path
#cap = cv2.VideoCapture(0)

#while True:
    
#    ret, frame = cap.read()
#    # Crop images from the frame
#    image_boxes = crop_image(frame)
#    # Perform any required preprocessing on the frame before prediction
#    for box in image_boxes:
#        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Display the frame
#   cv2.imshow("Fruit Detection", frame)

    # Assuming your model expects a certain input shape, resize and normalize the image
#    resized_frame = cv2.resize(frame, (224, 224))
#    normalized_frame = resized_frame / 255.0  # Normalize pixel values to be between 0 and 1
#    input_data = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
    
    # Make prediction
#    prediction = emotion_model.predict(input_data)
#    predicted_class = np.argmax(prediction)
    
    # Get the corresponding card label from the dictionary
#    fruit_label = fruit_dictionary[predicted_class]
    
    # Display the card label on the frame
#    cv2.putText(frame, str(fruit_label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
#    cv2.imshow("Fruit Prediction", frame)
    
    # Exit when 'q' is pressed
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

# Release the video capture object
#cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

