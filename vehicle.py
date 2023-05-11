import cv2
import numpy as np

# Web Camera
cap = cv2.VideoCapture('video.mp4')
min_width_rectangle = 80
min_height_rectangle = 80

# To give a line in video so that we can ount the vehcile
count_line_postion = 550


# Initialize Substructor:
algo = cv2.bgsegm.createBackgroundSubtractorMOG() # This alogo is used subtract the background and only detect the main object


# To counts the vehicle
def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

detect = []
offset = 6 # Allowable error between pixel
counter = 0


while True:
    ret,frame1 = cap.read()
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3,3), 5 )
    # Applying on each and every frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)  # to give shape Multichanel or Multitype image 
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #
    
    cv2.line(frame1,(25,count_line_postion),(1200,count_line_postion),(255,127,0),3)
    
    
    # to give rectange shape in vechile
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_rectangle) and (h>=min_height_rectangle)
        if not validate_counter:
            continue
        
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),2)  # { (0,0,255) = color code of green , 2 = thickness of rectangle }
        cv2.putText(frame1,"VEHICLE: "+str(counter),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,244,0),5)
  
           
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,255,0),-1)

        # To count in window:
        for (x,y) in detect:
            if y<(count_line_postion + offset) and y>(count_line_postion - offset):
                counter+=1
                cv2.line(frame1,(25,count_line_postion),(1200,count_line_postion),(0,127,255),3) # when vechile will cross the line it will change the line colour
                detect.remove((x,y))
                print("Vehicle counter: "+str(counter))
    
    
    # to put text in window
    cv2.putText(frame1,"VEHICLE COUNTER: "+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
    
    
    
    #cv2.imshow("Detor Original",dilatada)
    cv2.imshow("Video Original",frame1)

    if cv2.waitKey(1) == 27:
        break

#cv2.release()  
cap.release()  
cv2.destroyAllWindows()
