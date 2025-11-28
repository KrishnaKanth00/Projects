import cv2

template =cv2.imread('Screenshot 2025-11-28 113519.png',0)
h,w = template.shape

methods = [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED,cv2.TM_CCOEFF,cv2.TM_CCOEFF_NORMED,cv2.TM_CCORR,cv2.TM_CCORR_NORMED]

for method in methods:
    img = cv2.resize(cv2.imread('WIN_20251127_11_02_34_Pro.jpg'), (0, 0), fx=0.5, fy=0.5)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(img2,template,method)
    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(result)

    if method in [cv2.TM_SQDIFF_NORMED,cv2.TM_SQDIFF]:
        start = min_loc
    else:
        start = max_loc

    end = (start[0] + w,start[1] + h)
    img2 = cv2.rectangle(img,start,end,(0,0,255),4)

    cv2.imshow('result',img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




