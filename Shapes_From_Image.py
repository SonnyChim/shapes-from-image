from math import ceil
import cv2 as cv
import numpy as np
import time

groupsizerectangles = 100
groupsizetriangles = 0
groupsizecircles = 0
groupsizetotal = groupsizerectangles + groupsizetriangles + groupsizecircles
grouptoppercent = 5
grouptop = ceil(grouptoppercent / 100 * groupsizetotal)
children = groupsizetotal//grouptop
evolvegroupsize = children * grouptop
generations = 10
targetshapes = 100
targetdiff = 0000
randfactor = 1



orig_img = cv.imread("rcimg.png")
if orig_img is None:
    print('Could not open or find the image')
    exit(0)


weightmap = None
#weightmap = cv.imread("")

print(orig_img.shape)
imgy = orig_img.shape[0]
imgx = orig_img.shape[1]

def randrectnumbers():
    x1 = np.random.randint(0,imgx)
    x2 = np.random.randint(x1+1,imgx+1)
    y1 = np.random.randint(0,imgy)
    y2 = np.random.randint(y1+1,imgy+1)
    return np.array([x1,x2,y1,y2])

def randtrinumbers():
    x1 = np.random.randint(0,imgx)
    x2 = np.random.randint(0,imgx)
    x3 = np.random.randint(0,imgx)
    y1 = np.random.randint(0,imgy)
    y2 = np.random.randint(0,imgy)
    y3 = np.random.randint(0,imgy)
    return np.array([[x1,y1],[x2,y2],[x3,y3]],dtype=np.int32)

def randcirnumbers():
    x = np.random.randint(0,imgx)
    y = np.random.randint(0,imgy)
    r = np.random.randint(0,np.maximum(imgx,imgy))
    return np.array([x,y,r])

def randvalues(value,shape,factor):
    outputvalue = np.zeros(9)
    if shape == "r":
        for i in range(4):
            outputvalue[i] = value[i] + np.random.randint(-np.ceil(np.minimum(imgx,imgy)*factor),np.ceil(np.minimum(imgx,imgy)*factor))
            if outputvalue[i] < 0:
                outputvalue[i] = 0
            if outputvalue[i] > imgx - 1 and i == 0:
                outputvalue[i] = imgx - 1
            if outputvalue[i] > imgx and i == 1:
                outputvalue[i] = imgx
            if outputvalue[i] > imgy - 1 and i == 2:
               outputvalue[i] = imgy - 1
            if outputvalue[i] > imgy and i == 3:
                outputvalue[i] = imgy
        if outputvalue[2] > outputvalue[3]:
            outputvalue[2],outputvalue[3] = outputvalue[3],outputvalue[2]
        if outputvalue[0] > outputvalue[1]:
            outputvalue[0],outputvalue[1] = outputvalue[1],outputvalue[0]
    elif shape == "t":
        for i in range(6):
            outputvalue[i] = value[i] + np.random.randint(0,np.ceil(np.minimum(imgx,imgy)*factor))
            if outputvalue[i] < 0:
                outputvalue[i] = 0
            if outputvalue[i] > imgx and i in (0,2,4):
                outputvalue[i] = imgx
            if outputvalue[i] > imgy and i in (1,3,5):
                outputvalue[i] = imgy
    elif shape == "c":
        for i in range(3):
            outputvalue[i] = value[i] + np.random.randint(0,np.ceil(np.minimum(imgx,imgy)*factor))
            if outputvalue[i] < 0:
                outputvalue[i] = 0
            if outputvalue[i] > imgx and i == 0:
                outputvalue[i] = imgx
            if outputvalue[i] > imgy and i == 1:
                outputvalue[i] = imgy
    return outputvalue

def drawrect(img,values):
    cv.rectangle(img,(int(values[0]),int(values[2])),(int(values[1]),int(values[3])),(int(values[4]),int(values[5]),int(values[6])),-1)

def drawtri(img,values):
    cv.fillPoly(img,[np.array([[values[0],values[1]],[values[2],values[3]],[values[4],values[5]]],dtype=np.int32)],(int(values[6]),int(values[7]),int(values[8])))

def drawcir(img,values):
    cv.circle(img,(int(values[0]),int(values[1])),int(values[2]),(int(values[3]),int(values[4]),int(values[5])),-1)

if weightmap == None:
    def gen0(img,diffold):
        groupval = np.zeros((groupsizetotal,9))
        groupdiff = np.zeros(groupsizetotal)
        shape = np.empty(groupsizetotal,dtype=str)
        for i in range(groupsizerectangles):
            geoimg = img.copy()
            coords = randrectnumbers()
            values = np.append(coords,np.mean(orig_img[coords[2]:coords[3]+1,coords[0]:coords[1]+1],axis = (0,1)))
            shapediffbefore = np.sum(np.abs(np.subtract(orig_img[coords[2]:coords[3]+1,coords[0]:coords[1]+1],geoimg[coords[2]:coords[3]+1,coords[0]:coords[1]+1],dtype = np.int16)))
            drawrect(geoimg,values)
            shapediffafter = np.sum(np.abs(np.subtract(orig_img[coords[2]:coords[3]+1,coords[0]:coords[1]+1],geoimg[coords[2]:coords[3]+1,coords[0]:coords[1]+1],dtype = np.int16)))
            diff = diffold - shapediffbefore + shapediffafter
            groupval[i] = np.append(values,[0,0])
            groupdiff[i] = diff
            shape[i] = "r"
        for i in range(groupsizetriangles):
            geoimg = img.copy()
            mask = np.zeros((imgy, imgx, 3),dtype=np.uint8)
            coords = randtrinumbers()
            cv.fillPoly(mask,[coords],(1,1,1))
            values = np.append(coords,np.sum(orig_img*mask,axis = (0,1))/np.count_nonzero(mask)*3)
            drawtri(geoimg,values)
            diff = np.sum(np.abs(np.subtract(orig_img,geoimg,dtype = np.int16)))
            groupval[i+groupsizerectangles] = values
            groupdiff[i+groupsizerectangles] = diff
            shape[i+groupsizerectangles] = "t"
        for i in range(groupsizecircles):
            geoimg = img.copy()
            mask = np.zeros((imgy, imgx, 3),dtype=np.uint8)
            coords = randcirnumbers()
            cv.circle(mask,(coords[0],coords[1]),coords[2],(1,1,1),-1)
            values = np.append(coords,np.sum(orig_img*mask,axis = (0,1))/np.count_nonzero(mask)*3)
            drawcir(geoimg,values)
            diff = np.sum(np.abs(np.subtract(orig_img,geoimg,dtype = np.int16)))
            groupval[i+groupsizerectangles+groupsizetriangles] = np.append(values,[0,0,0])
            groupdiff[i+groupsizerectangles+groupsizetriangles] = diff
            shape[i+groupsizerectangles+groupsizetriangles] = "c"
        return groupval,groupdiff,shape
else:
    def gen0(img,diffold):
        groupval = np.zeros((groupsizetotal,9))
        groupdiff = np.zeros(groupsizetotal)
        shape = np.empty(groupsizetotal,dtype=str)
        for i in range(groupsizerectangles):
            geoimg = img.copy()
            coords = randrectnumbers()
            values = np.append(coords,np.mean(orig_img[coords[2]:coords[3]+1,coords[0]:coords[1]+1],axis = (0,1)))
            shapediffbefore = np.sum(np.abs(np.subtract(orig_img[coords[2]:coords[3]+1,coords[0]:coords[1]+1],geoimg[coords[2]:coords[3]+1,coords[0]:coords[1]+1],dtype = np.int16)) * weightmap[coords[2]:coords[3]+1,coords[0]:coords[1]+1])
            drawrect(geoimg,values)
            shapediffafter = np.sum(np.abs(np.subtract(orig_img[coords[2]:coords[3]+1,coords[0]:coords[1]+1],geoimg[coords[2]:coords[3]+1,coords[0]:coords[1]+1],dtype = np.int16)) * weightmap[coords[2]:coords[3]+1,coords[0]:coords[1]+1])
            diff = diffold - shapediffbefore + shapediffafter
            groupval[i] = np.append(values,[0,0])
            groupdiff[i] = diff
            shape[i] = "r"
        for i in range(groupsizetriangles):
            geoimg = img.copy()
            mask = np.zeros((imgy, imgx, 3),dtype=np.uint8)
            coords = randtrinumbers()
            cv.fillPoly(mask,[coords],(1,1,1))
            values = np.append(coords,np.sum(orig_img*mask,axis = (0,1))/np.count_nonzero(mask)*3)
            drawtri(geoimg,values)
            diff = np.sum(np.abs(np.subtract(orig_img,geoimg,dtype = np.int16)) * weightmap)
            groupval[i+groupsizerectangles] = values
            groupdiff[i+groupsizerectangles] = diff
            shape[i+groupsizerectangles] = "t"
        for i in range(groupsizecircles):
            geoimg = img.copy()
            mask = np.zeros((imgy, imgx, 3),dtype=np.uint8)
            coords = randcirnumbers()
            cv.circle(mask,(coords[0],coords[1]),coords[2],(1,1,1),-1)
            values = np.append(coords,np.sum(orig_img*mask,axis = (0,1))/np.count_nonzero(mask)*3)
            drawcir(geoimg,values)
            diff = np.sum(np.abs(np.subtract(orig_img,geoimg,dtype = np.int16)) * weightmap)
            groupval[i+groupsizerectangles+groupsizetriangles] = np.append(values,[0,0,0])
            groupdiff[i+groupsizerectangles+groupsizetriangles] = diff
            shape[i+groupsizerectangles+groupsizetriangles] = "c"
        return groupval,groupdiff,shape

def mouseinput(event,x,y,flags,param):
    global rectinputcoords,drawing,x1,y1
    if drawing:
        if event == cv.EVENT_LBUTTONDOWN:
            x1 = np.abs(x)
            y1 = np.abs(y)
        if event == cv.EVENT_LBUTTONUP:
            rectinputcoords = (np.minimum(x1,np.maximum(x,0)),np.maximum(x1,np.maximum(x,0)),np.minimum(y1,np.maximum(y,0)),np.maximum(y1,np.maximum(y,0)))
            drawing = False


if weightmap == None:
    def inputshape(img,diffold):
        cv.imshow("draw a rectangle", orig_img)
        global drawing,running
        drawing = True
        while drawing:
            if cv.waitKey(1) == ord("q"):
                running = False
                return 0,0,0
        values = np.append(rectinputcoords,np.mean(orig_img[rectinputcoords[2]:rectinputcoords[3]+1,rectinputcoords[0]:rectinputcoords[1]+1],axis = (0,1)))
        geoimg = img.copy()
        shapediffbefore = np.sum(np.abs(np.subtract(orig_img[rectinputcoords[2]:rectinputcoords[3]+1,rectinputcoords[0]:rectinputcoords[1]+1],geoimg[rectinputcoords[2]:rectinputcoords[3]+1,rectinputcoords[0]:rectinputcoords[1]+1],dtype = np.int16)))
        drawrect(geoimg,values)
        shapediffafter = np.sum(np.abs(np.subtract(orig_img[rectinputcoords[2]:rectinputcoords[3]+1,rectinputcoords[0]:rectinputcoords[1]+1],geoimg[rectinputcoords[2]:rectinputcoords[3]+1,rectinputcoords[0]:rectinputcoords[1]+1],dtype = np.int16)))
        diff = diffold - shapediffbefore + shapediffafter
        return (np.append(values,(0,0)),),(diff,),("r",)

else:
    def inputshape(img,diffold):
        cv.imshow("draw a rectangle", orig_img)
        global drawing,running
        drawing = True
        while drawing:
            if cv.waitKey(1) == ord("q"):
                running = False
                return 0,0,0
        values = np.append(rectinputcoords,np.mean(orig_img[rectinputcoords[2]:rectinputcoords[3]+1,rectinputcoords[0]:rectinputcoords[1]+1],axis = (0,1)))
        geoimg = img.copy()
        shapediffbefore = np.sum(np.abs(np.subtract(orig_img[rectinputcoords[2]:rectinputcoords[3]+1,rectinputcoords[0]:rectinputcoords[1]+1],geoimg[rectinputcoords[2]:rectinputcoords[3]+1,rectinputcoords[0]:rectinputcoords[1]+1],dtype = np.int16)) * weightmap[rectinputcoords[2]:rectinputcoords[3]+1,rectinputcoords[0]:rectinputcoords[1]+1])
        drawrect(geoimg,values)
        shapediffafter = np.sum(np.abs(np.subtract(orig_img[rectinputcoords[2]:rectinputcoords[3]+1,rectinputcoords[0]:rectinputcoords[1]+1],geoimg[rectinputcoords[2]:rectinputcoords[3]+1,rectinputcoords[0]:rectinputcoords[1]+1],dtype = np.int16)) * weightmap[rectinputcoords[2]:rectinputcoords[3]+1,rectinputcoords[0]:rectinputcoords[1]+1])
        diff = diffold - shapediffbefore + shapediffafter
        return (np.append(values,(0,0)),),(diff,),("r",)

if weightmap == None:
    def genx(inputvalues,shape,img,diffold):
        groupdiff = np.zeros(evolvegroupsize)
        outputvalues = inputvalues
        for i in range(evolvegroupsize):
            if shape[i] == "r":
                geoimg = img.copy()
                values = np.append(inputvalues[i][0:4],np.mean(orig_img[inputvalues[i][2]:inputvalues[i][3]+1,inputvalues[i][0]:inputvalues[i][1]+1],axis = (0,1)))
                shapediffbefore = np.sum(np.abs(np.subtract(orig_img[inputvalues[i][2]:inputvalues[i][3]+1,inputvalues[i][0]:inputvalues[i][1]+1],geoimg[inputvalues[i][2]:inputvalues[i][3]+1,inputvalues[i][0]:inputvalues[i][1]+1],dtype = np.int16)))
                drawrect(geoimg,values)
                shapediffafter = np.sum(np.abs(np.subtract(orig_img[inputvalues[i][2]:inputvalues[i][3]+1,inputvalues[i][0]:inputvalues[i][1]+1],geoimg[inputvalues[i][2]:inputvalues[i][3]+1,inputvalues[i][0]:inputvalues[i][1]+1],dtype = np.int16)))
                groupdiff[i] = diffold - shapediffbefore + shapediffafter
                outputvalues[i][4:7] = values[4:7]
            elif shape[i] == "t":
                geoimg = img.copy()
                mask = np.zeros((imgy, imgx, 3),dtype=np.uint8)
                cv.fillPoly(mask,[np.array([inputvalues[i][0:2],inputvalues[i][2:4],inputvalues[i][4:6]],dtype = np.int32)],(1,1,1))
                maskvalue = np.count_nonzero(mask)
                if maskvalue:
                    values = np.append(inputvalues[i][0:6],np.sum(orig_img*mask,axis = (0,1))/maskvalue*3)
                else:
                    values = np.append(inputvalues[i][0:6],(0,0,0))
                drawtri(geoimg,values)
                groupdiff[i] = np.sum(np.abs(np.subtract(orig_img,geoimg,dtype = np.int16)))
                outputvalues[i][6:9] = values[6:9]
            elif shape[i] == "c":
                geoimg = img.copy()
                mask = np.zeros((imgy, imgx, 3),dtype=np.uint8)
                cv.circle(mask,(inputvalues[i][0],inputvalues[i][1]),inputvalues[i][2],(1,1,1),-1)
                values = np.append(inputvalues[i][0:3],np.sum(orig_img*mask,axis = (0,1))/np.count_nonzero(mask)*3)
                drawcir(geoimg,values)
                groupdiff[i] = np.sum(np.abs(np.subtract(orig_img,geoimg,dtype = np.int16)))
                outputvalues[i][3:6] = values[3:6]
        return groupdiff
else:
    def genx(inputvalues,shape,img,diffold):
        groupdiff = np.zeros(evolvegroupsize)
        outputvalues = inputvalues
        for i in range(evolvegroupsize):
            if shape[i] == "r":
                geoimg = img.copy()
                values = np.append(inputvalues[i][0:4],np.mean(orig_img[inputvalues[i][2]:inputvalues[i][3]+1,inputvalues[i][0]:inputvalues[i][1]+1],axis = (0,1)))
                shapediffbefore = np.sum(np.abs(np.subtract(orig_img[inputvalues[i][2]:inputvalues[i][3]+1,inputvalues[i][0]:inputvalues[i][1]+1],geoimg[inputvalues[i][2]:inputvalues[i][3]+1,inputvalues[i][0]:inputvalues[i][1]+1],dtype = np.int16)) * weightmap[inputvalues[i][2]:inputvalues[i][3]+1,inputvalues[i][0]:inputvalues[i][1]+1])
                drawrect(geoimg,values)
                shapediffafter = np.sum(np.abs(np.subtract(orig_img[inputvalues[i][2]:inputvalues[i][3]+1,inputvalues[i][0]:inputvalues[i][1]+1],geoimg[inputvalues[i][2]:inputvalues[i][3]+1,inputvalues[i][0]:inputvalues[i][1]+1],dtype = np.int16)) * weightmap[inputvalues[i][2]:inputvalues[i][3]+1,inputvalues[i][0]:inputvalues[i][1]+1])
                groupdiff[i] = diffold - shapediffbefore + shapediffafter
                outputvalues[i][4:7] = values[4:7]
            elif shape[i] == "t":
                geoimg = img.copy()
                mask = np.zeros((imgy, imgx, 3),dtype=np.uint8)
                cv.fillPoly(mask,[np.array([inputvalues[i][0:2],inputvalues[i][2:4],inputvalues[i][4:6]],dtype = np.int32)],(1,1,1))
                maskvalue = np.count_nonzero(mask)
                if maskvalue:
                    values = np.append(inputvalues[i][0:6],np.sum(orig_img*mask,axis = (0,1))/maskvalue*3)
                else:
                    values = np.append(inputvalues[i][0:6],(0,0,0))
                drawtri(geoimg,values)
                groupdiff[i] = np.sum(np.abs(np.subtract(orig_img,geoimg,dtype = np.int16)) * weightmap)
                outputvalues[i][6:9] = values[6:9]
            elif shape[i] == "c":
                geoimg = img.copy()
                mask = np.zeros((imgy, imgx, 3),dtype=np.uint8)
                cv.circle(mask,(inputvalues[i][0],inputvalues[i][1]),inputvalues[i][2],(1,1,1),-1)
                values = np.append(inputvalues[i][0:3],np.sum(orig_img*mask,axis = (0,1))/np.count_nonzero(mask)*3)
                drawcir(geoimg,values)
                groupdiff[i] = np.sum(np.abs(np.subtract(orig_img,geoimg,dtype = np.int16)) * weightmap)
                outputvalues[i][3:6] = values[3:6]
        return groupdiff

def evolve(gen,img,inputvalues,shape,diffold):
    if gen == 0:
        if manualmode:
            return inputshape(img,diffold)
        else:
            a,b,c = gen0(img,diffold)
            indeces = b.argsort()
            return a[indeces][0:grouptop],b[indeces][0:grouptop],c[indeces][0:grouptop]
    else:
        outputvalues = np.zeros((evolvegroupsize,9))
        outputshape = np.empty(evolvegroupsize,dtype = str)
        for x in range(grouptop):
            for y in range(children-1):
                outputvalues[y + x * children] = randvalues(inputvalues[x],shape[x],randfactor/gen)
                outputshape[y + x * children] = shape[x]
            outputvalues[children - 1 + x * children] = inputvalues[x]
            outputshape[children - 1 + x * children] = shape[x]
        outputvalues = outputvalues.astype(np.uint16)
        outputdiff = genx(outputvalues,outputshape,img,diffold)
        indeces = outputdiff.argsort()
        return outputvalues[indeces][0:grouptop],outputdiff[indeces][0:grouptop],outputshape[indeces][0:grouptop]

def createshape(diffold,img):
    global txtout,running
    gen = 0
    values,shape = 0,0
    while True:
        for i in range(generations):
            values,diff,shape = evolve(gen,img,values,shape,diffold)
            imgtest = imgout.copy()
            if cv.waitKey(1) == ord("q") or not running:
                running = False
                return 0
            if shape[0] == "r":
                drawrect(imgtest,values[0])
            elif shape[0] == "t":
                drawtri(imgtest,values[0])
            elif shape[0] == "c":
                drawcir(imgtest,values[0])
            cv.imshow("evolve",imgtest)
            gen += 1
        if diff[0]<diffold:
            if shape[0] == "r":
                drawrect(imgout,values[0])
                values = values[0].astype(np.uint16)
                txtout += f"r({values[0]},{values[2]},{values[1]-values[0]},{values[3]-values[2]},{values[4]},{values[5]},{values[6]})\n"
            elif shape[0] == "t":
                drawtri(imgout,values[0])
                values = values[0].astype(np.uint16)
                txtout += f"t({values[0]},{values[1]},{values[2]},{values[3]},{values[4]},{values[5]},{values[6]},{values[7]},{values[8]})\n"
            elif shape[0] == "c":
                drawcir(imgout,values[0])
                values = values[0].astype(np.uint16)
                txtout += f"c({values[0]},{values[1]},{values[2]},{values[3]},{values[4]},{values[5]})\n"
            diffold = diff[0]
            return diffold
        else:
            gen = 0

def generateimage():
    global running
    if weightmap == None:
        diffold = np.sum(np.abs(np.subtract(orig_img,imgout,dtype = np.int16)))
    else:
        diffold = np.sum(np.abs(np.subtract(orig_img,imgout,dtype = np.int16)) * weightmap)
    shapes = 0
    while ((diffold/(imgx*imgy*3)) > targetdiff or shapes < targetshapes) and running == True:
        print(f"shape {shapes}, difference: {diffold/(imgx*imgy*3)}")
        diffold = createshape(diffold,imgout)
        shapes += 1
        cv.imshow("out",imgout)
        cv.imshow("difference",np.abs(np.subtract(orig_img,imgout,dtype = np.int16)).astype(np.uint8))
        if cv.waitKey(1) == ord("q"):
            running = False
            break
    #print(txtout)
    if running:
        print(f"shape {shapes}, difference: {diffold/(imgx*imgy*3)}")

# manualmode = True
manualmode = False
if manualmode:
    groupsizerectangles = 100
    groupsizetotal = groupsizerectangles
    grouptop = 1
    children = groupsizetotal//grouptop
    evolvegroupsize = children * grouptop
    randfactor = 1/10
    rectinputcoords = None
    cv.namedWindow("draw a rectangle")
    cv.setMouseCallback("draw a rectangle", mouseinput)

running = True
imgout = np.zeros((imgy, imgx, 3),dtype=np.uint8)
cv.rectangle(imgout,(0,0),(imgx,imgy),np.mean(orig_img,axis = (0,1)),-1)
print(np.mean(orig_img,axis = (0,1)))
txtout = ""
txt = open("shapes.txt","w")
cv.imshow("original image",orig_img)
start = time.perf_counter_ns()
generateimage()
end = time.perf_counter_ns()
print(f"Done in {(end-start)/10**9}")
txt.write(txtout)
txt.close
cv.imwrite("out.png",imgout)
cv.waitKey()
