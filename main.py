# Import das bibliotecas OpenCV e Mediapipe para o código.
import cv2
import mediapipe as mp

# Criação dos objetos para desenhar as mãos detectadas pela imagem.
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Capturar imagens da webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode = False, # Falso por que é um video
    max_num_hands = 2, # Número max de mãos detectadas 
    min_detection_confidence = 0.5) as hands:

    # Loop infinito para capturar imagens da webcam
    while True:
        ret , frame = cap.read()
        # Verifica se a captura de vídeo foi bem-sucedida.
        if ret == False:
            break
        
        # Obtém as dimenções da imagem
        height, width, _ = frame.shape
        # Inverte a imagem horizontalmente
        frame = cv2.flip(frame,1)
        # Converte a imagem de BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processa a imagem para detectar as mãos
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks is not None:

            # Define a cor das conexões das linhas e a cor dos círculos na detecção.
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(156 ,48 ,64 ), thickness=4))

        cv2.imshow('Transmissão da Webcam', frame)

        # Aguarda a tecla ‘q’ ser pressionada para sair do loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()