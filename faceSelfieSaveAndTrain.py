import cv2
import numpy as np

# initialize camera
cap = cv2.VideoCapture(0)

 # face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


dataset_path = './data/'
face_data = []
skip = 0
file_name =input("Enter the name of the person ")

while True :
	ret,frame = cap.read()

	if ret==False:
		continue

	gframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	

	faces = face_cascade.detectMultiScale(gframe,1.3,5)
	if len(faces)==0:
		continue
		
	faces = sorted(faces,key=lambda f:f[2]*f[3])

	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(gframe,(x,y),(x+w,y+h),(0,255,255),2)

		offset = 10
		face_section = gframe[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip += 1
		if skip%10==0:
			face_data.append(face_section)
			print(gframe.shape)



	cv2.imshow("face detector",gframe)
	key_pressed = cv2.waitKey(1) & 0XFF
	if key_pressed == ord('q'):
		break




# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
print("bfr",face_data.shape)
face_data = face_data.reshape((face_data.shape[0],-1))
print("aftr",face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()	
