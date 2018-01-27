import numpy as np
import cv2

# Key
A = 'A'
B = 'B'
C = 'C'
D = 'D'
E = 'E'
F = 'F'
G = 'G'
H = 'H'
I = 'I'
J = 'J'
K = 'K'

# Classifiers
A_cascade = cv2.CascadeClassifier(A + '_25_HAAR.xml')
B_cascade = cv2.CascadeClassifier(B + '_25_HAAR.xml')
C_cascade = cv2.CascadeClassifier(C + '_25_HAAR.xml')
D_cascade = cv2.CascadeClassifier(D + '_35_HAAR_newNegs.xml')
E_cascade = cv2.CascadeClassifier(E + '_31_HAAR.xml')
F_cascade = cv2.CascadeClassifier(F + '_26_HAAR.xml')
G_cascade = cv2.CascadeClassifier(G + '_34_HAAR_newNegs.xml')
H_cascade = cv2.CascadeClassifier(H + '_34_HAAR_newNegs.xml')
I_cascade = cv2.CascadeClassifier(I + '_35_HAAR.xml')
J_cascade = cv2.CascadeClassifier(J + '_26_HAAR.xml')
K_cascade = cv2.CascadeClassifier(K + '_28_HAAR.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()

    # Cropped Frame
    width = int(cap.get(3)+0.5)
    height = int(cap.get(4)+0.5)
    x_ = width//4
    y_ = height//7
    w_ = width//3
    h_ = width//3
    cv2.rectangle(img, (x_-2, y_-2), (x_ + w_+2, y_ + h_+2), (0, 255, 0), 1)
    crop_y = y_ + w_
    crop_x = x_ + w_
    cropped_frame = img[y_+2:crop_y-2, x_+2: crop_x-2, :]

    # Convert to Gray Scale
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

    a = A_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    b = B_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    c = C_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    d = D_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    e = E_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    f = F_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    g = G_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    h = H_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    i = I_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    j = J_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    k = K_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20)
    )
    
    # Find Letter
    #   - if(1 object found)
    if(len(a) == 1):
        cv2.putText(img, A. format(1), (x_, y_ - 10,), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 100, 200), 2)

    elif(len(b) == 1):
        cv2.putText(img, B.format(1), (x_, y_ - 10,), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 100, 200), 2)

    elif(len(c) == 1):
        cv2.putText(img, C.format(1), (x_, y_ - 10,), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 100, 200), 2)

    elif(len(d) == 1):
        cv2.putText(img, D.format(1), (x_, y_ - 10,), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 100, 200), 2)

    elif(len(e) == 1):
        cv2.putText(img, E.format(1), (x_, y_ - 10,), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 100, 200), 2)
    
    elif(len(f) == 1):
        cv2.putText(img, F.format(1), (x_, y_ - 10,), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 100, 200), 2)

    elif(len(g) == 1):
        cv2.putText(img, G.format(1), (x_, y_ - 10,), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 100, 200), 2)

    elif(len(h) == 1):
        cv2.putText(img, H.format(1), (x_, y_ - 10,), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 100, 200), 2)

    elif(len(i) == 1):
        cv2.putText(img, I.format(1), (x_, y_ - 10,), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 100, 200), 2)

    elif(len(j) == 1):
        cv2.putText(img, J.format(1), (x_, y_ - 10,), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 100, 200), 2)

    elif(len(k) == 1):
        cv2.putText(img, K.format(1), (x_, y_ - 10,), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (50, 100, 200), 2)

    # Show Video
    cv2.imshow('img',img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
