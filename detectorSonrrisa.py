#importamos cv
import cv2

#Cargamos los clasificadores para caras y sonrrisas
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

#Tomamos la webcam
webcam = cv2.VideoCapture(0)


while True :
    #Leemos la primer imagen, obtenemos un OK y la imagen.
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break
    #Cambiamos a blanco y negro para no procesar color en los clasificadores
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Pasamos la imagen por el detector de caras y sonrrisas, obtenemos los puntos donde hay caras o sonrrisas.
    faces = face_detector.detectMultiScale(frame_grayscale)
    smiles = smile_detector.detectMultiScale(frame_grayscale, scaleFactor= 1.7, minNeighbors= 20)

    # tenemos las coordenadas x e y de los puntos junto con el ancho y alto del rectangulo.
    for (x, y, w, h) in faces :
        #Escribimos un rectangulo con esos puntos.
        cv2.rectangle(frame,(x, y), (x+w, y+h), (100, 200, 50), 4)
        #Guardamos el rectangulo con la cara de la imagen.
        the_face = frame[y:y+h, x:x+w]
        #Cambiamos a blanco y negro
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor= 1.7, minNeighbors= 20)

        #Buscamos sonrrisas en la cara
        for (x_, y_, w_, h_) in smiles :
            #Dibujamos el rectangulo de la sonrrisa!
            cv2.rectangle(the_face,(x_, y_), (x_+w_, y_+h_), (50, 50, 200), 4)
        #Escribimos "sonrriendo!"
        if len(smiles) > 0 : 
            cv2.putText(frame, 'Sonrriendo!', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    cv2.imshow ('Detector de sonrrisas', frame)

    #Hacemos que cada 1 ms continue y entre en el loop tomando otra imagen... Simula tiempo real.
    cv2.waitKey(1)

#Liberamos camara y cv2
webcam.release()
cv2.destroyAllWindows()
