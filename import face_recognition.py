import face_recognition
import cv2
import os
import numpy as np

# Step 1: Load known face encodings, names, and associated places
def load_known_faces_and_places(known_faces_dir):
    known_encodings = []
    known_names = []
    known_places = {}

    # Example: Map names to places (can be extended with a database or file)
    name_to_place = {
        "Hasanul": "Prof.",
        "Muhammad": "Prof.",
        "Abu": "Prof.",
        "Kamrul": "Prof.",
        "Mohammad": "Vc",
        "lal person": "lal group"
    }

    # Iterate over each subdirectory in the known_faces directory
    for name_dir in os.listdir(known_faces_dir):
        name_path = os.path.join(known_faces_dir, name_dir)
        if not os.path.isdir(name_path):
            continue

        name_encodings = []

        for file_name in os.listdir(name_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(name_path, file_name)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)

                if encodings:
                    name_encodings.append(encodings[0])
                    print(f"[INFO] Loaded encoding for {name_dir} from {file_name}")
                else:
                    print(f"[WARNING] No faces found in {file_name}. Skipping.")

        # Average the encodings for better generalization
        if name_encodings:
            averaged_encoding = np.mean(name_encodings, axis=0)
            known_encodings.append(averaged_encoding)
            known_names.append(name_dir)
            # Associate the name with a place
            known_places[name_dir] = name_to_place.get(name_dir, "Unknown Location")

    return known_encodings, known_names, known_places

# Step 2: Recognize faces and display their names and places
def recognize_faces_with_places(image_path, known_encodings, known_names, known_places, annotate=True, save_output=True):
    input_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(input_image)
    face_encodings = face_recognition.face_encodings(input_image, face_locations)

    print(f"[INFO] Found {len(face_locations)} face(s) in the image.")

    input_image_cv2 = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    for i, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings), start=1):
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        place = "Unknown Location"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = None
        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                place = known_places.get(name, "Unknown Location")

        print(f"[INFO] Face {i}: {name} ({place}) at Location (Top: {top}, Right: {right}, Bottom: {bottom}, Left: {left})")

        # Draw a rectangle around the face
        frame_color = (0, 255, 0)  # Green
        cv2.rectangle(input_image_cv2, (left, top), (right, bottom), frame_color, 2)

        # Annotate with name and place
        label = f"{name} - {place}"
        cv2.putText(input_image_cv2, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    window_name = 'Recognized Faces and Places'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.imshow(window_name, input_image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_output:
        output_path = os.path.splitext(image_path)[0] + "_recognized_with_places.jpg"
        cv2.imwrite(output_path, input_image_cv2)
        print(f"[INFO] Saved annotated image to {output_path}")

# Step 3: Main program
if __name__ == "__main__":
    known_faces_directory = "known_faces"
    test_image_path = "test.jpg"

    if not os.path.isdir(known_faces_directory):
        print(f"[ERROR] The directory '{known_faces_directory}' does not exist.")
        exit(1)

    if not os.path.isfile(test_image_path):
        print(f"[ERROR] The image file '{test_image_path}' does not exist.")
        exit(1)

    print("[INFO] Loading known faces and places...")
    known_encodings, known_names, known_places = load_known_faces_and_places(known_faces_directory)

    if not known_encodings:
        print("[ERROR] No known face encodings found. Exiting.")
        exit(1)

    print("[INFO] Recognizing faces in the test image...")
    recognize_faces_with_places(
        image_path=test_image_path, 
        known_encodings=known_encodings, 
        known_names=known_names, 
        known_places=known_places, 
        annotate=True, 
        save_output=True
    )
