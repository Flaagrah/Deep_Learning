from mainmodel import BOX_HEIGHT as BOX_HEIGHT
from mainmodel import BOX_WIDTH as BOX_WIDTH
from mainmodel import IMAGE_HEIGHT as IMAGE_HEIGHT
from mainmodel import IMAGE_WIDTH as IMAGE_WIDTH


CERTAINTY_THRESHOLD = 0.95

def hitOrMiss(flag):
    if flag > CERTAINTY_THRESHOLD:
        return True
    return False

def processResults(predictionsForOneImage, mode = 'MINIMUM_THRESHOLD'):
    #number of rows
    refBoxRows = IMAGE_HEIGHT/BOX_HEIGHT
    #number of columns
    refBoxColumns = IMAGE_WIDTH/BOX_WIDTH
    boxes = []
    
    largest = -1.0
    largestIndex = 0       
    
    for i in range(0, int(len(predictionsForOneImage)/5)):
        boxIndex = i*5
        boxFlag = predictionsForOneImage[boxIndex]
        if boxFlag > largest:
            largest = boxFlag
            largestIndex = boxIndex
        if  (mode == 'MINIMUM_THRESHOLD' and hitOrMiss(boxFlag)) or (mode == 'LARGEST' and (boxIndex + 6 > len(predictionsForOneImage))) :
            boxVerLoc = predictionsForOneImage[boxIndex+1]
            boxHorLoc = predictionsForOneImage[boxIndex+2]
            boxHeight = predictionsForOneImage[boxIndex+3]
            boxWidth = predictionsForOneImage[boxIndex+4]
            
            row = int(i/refBoxColumns)
            column = int(i % refBoxColumns)
            
            #Get the actual coordinates and height/width in 256*256 version of image.
            verCoord = ((row +boxVerLoc) * BOX_HEIGHT)
            horCoord = ((column+boxHorLoc) * BOX_WIDTH)
            height = boxHeight * IMAGE_HEIGHT
            width = boxWidth * IMAGE_WIDTH
            
            #Making sure that the boundaries are within the image.
            if verCoord < 0.0:
                verCoord = 0.0
            if horCoord < 0.0:
                horCoord = 0.0
            if (verCoord - (height / 2.0)) < 0.0:
                height = verCoord * 2.0
            if (horCoord - (width / 2.0)) < 0.0:
                width = horCoord * 2.0
            if (verCoord + (height / 2.0)) > 255.0:
                height = (255.0 - verCoord) * 2.0
            if (horCoord + (width / 2.0)) > 255.0:
                width = (255.0 - horCoord) * 2.0
            
            boxVals = []
            boxVals.append(boxFlag)
            boxVals.append(verCoord)
            boxVals.append(horCoord)
            boxVals.append(height)
            boxVals.append(width)
            boxes.append(boxVals)
    
    def coords(box):
        x1 = box[2] - (box[4]/2)
        x2 = box[2] + (box[4]/2)
        y1 = box[1] - (box[3]/2)
        y2 = box[1] + (box[3]/2)
        return x1, x2, y1, y2
    
    def getOverLap(start1, end1, start2, end2):
        startOverLap = -1
        endOverLap = -1
        if start1 >= start2 and start1 <= end2:
            startOverLap = start1
        elif start2 >= start1 and start2 <= end1:
            startOverLap = start2
        if startOverLap == -1:
            return 0
        if end1 <= end2:
            endOverLap = end1
        else:
            endOverLap = end2
        return endOverLap - startOverLap
    
    boxResults = []
    removeIndices = []
    checkedIndices = []
    for i in range(0, len(boxes)):
        removeIndices.append(False)
        checkedIndices.append(False)
    while True:
        numBoxes = len(boxes)
        checked = True
        maxBox = None
        maxBoxIndex = 0
        for i in range(0, numBoxes):
            box1 = boxes[i]
            if (maxBox is None or box1[0] > maxBox[0]) and removeIndices[i] == False and checkedIndices[i] == False:
                maxBox = box1
                #print("DFHIODHFIOHDI")
                maxBoxIndex = i
                checked = False
                
        if checked == True:
            break
        checkedIndices[maxBoxIndex] = True
        x11, x12, y11, y12 = coords(maxBox)
        #print(maxBox[0])
        #print(x11)
        #print(x12)
        #print(y11)
        #print(y12)
        
        i = 0
        for i in range(0, numBoxes):
            if i == maxBoxIndex or removeIndices[i] == True or checkedIndices[i] == True:
                continue
            #print("here")
            box1 = boxes[i]
            #print(box1[0])
            #print(maxBox[0])
            if box1[0] < maxBox[0]:
                #print("there")
                x21, x22, y21, y22 = coords(box1)
                #print(maxBox[0])
                #print(x21)
                #print(x22)
                #print(y21)
                #print(y22)
                xLap = getOverLap(x11, x12, x21, x22)
                yLap = getOverLap(y11, y12, y21, y22)
                lap = xLap*yLap
                IOU = lap / (maxBox[3]*maxBox[4] + box1[3]*box1[4] - lap)
                #print("lap")
                #print(maxBox[3]*maxBox[4])
                #print(box1[3]*box1[4])
                #print(xLap)
                #print(yLap)
                #print(lap)
                #print("IOU")
                #print(IOU)
                if IOU > 0.2:
                    removeIndices[i] = True
                    
    for i in range(0, len(boxes)):
        if not removeIndices[i]:
            box = boxes[i]
            box = box[1:]
            boxResults.append(box)
            
    return boxResults


def generateOutput(imgNames, imgPreds, testDims):
    names = []
    encoding = []
    for i in range(0, len(imgPreds)):
        img = imgPreds[i]
        dims = testDims[i]
        name = imgNames[i]
        
        boxResults = processResults(img)
        if len(boxResults) == 0:
            boxResults = processResults(img, 'LARGEST')
        print(name)
        print(len(boxResults))
        verDim = dims[0]
        horDim = dims[1]
        
        horMultiple = horDim/IMAGE_WIDTH
        verMultiple = verDim/IMAGE_HEIGHT
        
        traversedPixels = []
        added = False
        #Process each mask using the boxes.
        for j in range(0, len(boxResults)):
            runLength = ''
            box = boxResults[j]
            verCoord = int(box[0] * verMultiple)
            horCoord = int(box[1] * horMultiple)
            height = int(box[2] * verMultiple)
            width = int(box[3] * horMultiple)
            
            leftSide = horCoord - int(width/2)
            rightSide = horCoord + int(width/2)
            top = verCoord - int(height/2)
            bottom = verCoord + int(height/2)
            
            newAdditions = []
            #Encode the pixels for each mask
            for w in range(leftSide, rightSide):
                topPoint = int(w * verDim) + top + 1
                if topPoint < 1:
                    topPoint = 1
                bottomPoint = topPoint + height - 1
                lastPixel = horDim*verDim
                if bottomPoint > lastPixel:
                    bottomPoint = lastPixel
                
                pair = [topPoint, bottomPoint]
                addSegment(pair, traversedPixels, newAdditions)
            for a in range(0, len(newAdditions)):
                newAdd = newAdditions[a]
                newTop = newAdd[0]
                newBottom = newAdd[1]
                height = newBottom - newTop + 1
                runLength += ' ' + str(newTop) + ' ' + str(height)
                                        
            if len(runLength) > 1:
                runLength = runLength[1:]
            elif len(runLength) == 1 or len(runLength)==0:
                runLength = ''
            if not runLength == '':
                names.append(name)
                encoding.append(runLength)
                added = True
        if not added:
            names.append(name)
            encoding.append('1 1')
    return names, encoding


#Given top and bottom of segment, returns all segmentations.
def addSegment(segmentPair, traversedPixels, newAdditions):
    topPoint = segmentPair[0]
    bottomPoint = segmentPair[1]
    if topPoint > bottomPoint:
        return
    for t in range(0, len(traversedPixels)):
        pixelPair = traversedPixels[t]
        tTop = pixelPair[0]
        tBottom = pixelPair[1]
        #Changing the top and bottom point to avoid duplicating pixels.
        #If top point is among pixels already masked, move it beyond the already masked segment.
        if (topPoint >= tTop and topPoint <= tBottom):
            topPoint = tBottom + 1
            aSeg = [topPoint, bottomPoint]
            addSegment(aSeg, traversedPixels, newAdditions)
            return
        #If the bottom point is among pixels already masked, move it before the already masked segment.
        if (bottomPoint >= tTop and bottomPoint <= tBottom):
            bottomPoint = tTop - 1
            aSeg = [topPoint, bottomPoint]
            addSegment(aSeg, traversedPixels, newAdditions)
            return
        if (topPoint <= tTop and bottomPoint >= tBottom):
            aBottom = tTop - 1
            aTop = tBottom + 1
            seg1 = [topPoint, aBottom]
            seg2 = [aTop, bottomPoint]
            addSegment(seg1, traversedPixels, newAdditions)
            addSegment(seg2, traversedPixels, newAdditions)
            return
    newAdditions.append([topPoint, bottomPoint]) 
    traversedPixels.append([topPoint, bottomPoint])
    
