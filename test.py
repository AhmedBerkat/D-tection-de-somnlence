import cv2
import dlib
import numpy as np

# Charger le détecteur de visage et le prédicteur de points
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialiser la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la caméra.")
    exit()

# Variables pour suivre le temps de fermeture des yeux
closed_eye_duration = 0
eye_closed_threshold = 6  # Seuil en secondes pour considérer comme somnolent
eye_aspect_ratio_threshold = 0.25  # Seuil pour la fermeture des yeux
yawn_counter = 0  # Compteur pour les bâillements

def eye_aspect_ratio(eye):
    # Calculer le rapport d'aspect de l'œil
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Calculer le rapport d'aspect de la bouche
    A = np.linalg.norm(mouth[13] - mouth[19])  # Distance verticale entre les lèvres
    B = np.linalg.norm(mouth[14] - mouth[18])  # Distance verticale au centre de la bouche
    C = np.linalg.norm(mouth[12] - mouth[16])  # Distance horizontale entre les coins de la bouche
    mar = (A + B) / (2.0 * C)
    return mar

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : impossible de lire le cadre.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Points des yeux gauche et droit
        left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])

        # Calculer le rapport d'aspect des yeux
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Vérifier la fermeture des yeux
        if ear < eye_aspect_ratio_threshold:
            closed_eye_duration += 1  # Compter le temps en secondes
        else:
            closed_eye_duration = 0  # Réinitialiser si les yeux sont ouverts

        # Avertir si l'utilisateur est somnolent à cause de la fermeture des yeux
        if closed_eye_duration >= eye_closed_threshold:
            cv2.putText(frame, "Les Yeux fermes , Reveille-toi !", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Points de la bouche
        mouth = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])

        # Calculer le rapport d'aspect de la bouche
        mar = mouth_aspect_ratio(mouth)

        # Détection de bâillements avec seuil ajusté
        if mar > 0.4:  # Seuil pour la détection des bâillements (ajuster si nécessaire)
            yawn_counter += 1
            if yawn_counter > 1:
                cv2.putText(frame, " Baillement, Reveille-toi !", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,0), 2)
        else:
            yawn_counter = 0  # Réinitialiser le compteur

        # Dessiner les points de repère sur le visage
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Dessiner un cercle vert

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
