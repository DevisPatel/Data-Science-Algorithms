import import_ipynb
import glob
import os
import shutil
import cv2 as cv

image = cv.imread(r"C:\Users\patel\Documents\Job Documents\img1.jpg", -1)

gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
binary_img = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9)


binary_img = cv.bitwise_not(binary_img)


kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
dilated_img = cv.morphologyEx(binary_img, cv.MORPH_DILATE, kernel,iterations=1)


(cnts, _)= cv.findContours(dilated_img.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print("Number of countours = "+ str(len(cnts)))


sq_cnts = [] 

for cnt in cnts:
    approx = cv.approxPolyDP(cnt,0.01*cv.arcLength(cnt,True),True)
    
    if len(approx) == 4:
      (x, y, w, h) = cv.boundingRect(cnt)

      
      aspectRatio = float(w) / h
      print (aspectRatio,"Aspectratio") 
      if aspectRatio > 1.0 and aspectRatio <= 1.08:
          cv.drawContours(image,[cnt], 0, (0,255,0), 3)
          sq_cnts.append(cnt)

          for i in range(len(sq_cnts)):
            
            (x, y, w, h) = cv.boundingRect(sq_cnts[i])
            newimg = image[y:y+h,x:x+w] 

    
            cv.imwrite(str(i)+"\n \n \n \n \n \n \n \n \n \n \n \n \t\t\\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"+'.jpg',newimg)
            jpgfile=cv.imwrite(str(i)+"\n \n \n \n \n \n \n \n \n \n \n \n "+'.jpg',newimg)
            cv.imshow('img.jpg', newimg)
            
            for jpgfile in glob.iglob(os.path.join(r"C:\Users\patel\JpPrograms", "*.jpg")):
                shutil.copy(jpgfile,r'C:\Users\patel\Documents\ThinkTac')
            

cv.waitKey(0)
cv.destroyAllWindows()


