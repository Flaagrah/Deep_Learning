import numpy as np

from mainmodel import IMAGE_HEIGHT
from mainmodel import IMAGE_WIDTH


def processResults(output):
    allResults = []
    outputArray = np.array(output)
    outputArray = np.reshape(outputArray, (-1, IMAGE_HEIGHT, IMAGE_WIDTH))
    output = outputArray.tolist()
    for mask in output:
        result = [[0.0 if (element < 0.5) else 1.0 for element in row] for row in mask]
        allResults.append(result)
    
    results = np.array(allResults).astype(np.float64)
    return np.reshape(results, (-1, IMAGE_HEIGHT*IMAGE_WIDTH))

def generateOutput(masks):
    runLengths = []
    for i in range(0, len(masks)):
        img = masks[i]
        size = IMAGE_HEIGHT*IMAGE_WIDTH
        runLength = ""
        currLength = 0
        currIndex = 0
        print(i)
        for j in range(0, size):
            if img[j]==0.0 and not currLength==0:
                runLength += str(currIndex+1)+" "+str(currLength)+" "
                currLength=0
            elif img[j]==1.0 and currLength==0:
                currIndex=j
                currLength=1
            elif img[j]==1.0 and currLength>0:
                currLength += 1
                
        if currLength>0:
            runLength += str(currIndex+1)+" "+str(currLength)+" "
        if runLength=="":
            runLength="1 1"
        else:
            runLength = runLength[0:(len(runLength)-1)]
        runLengths.append(runLength)
    return runLengths