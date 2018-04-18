import pandas

def pixels(runLength):
    print("Run Length")
    run = runLength.split(' ')
    pixels = []
    for i in range(0, len(run)):
        if i%2 == 0:
            start = run[i]
            leng = run[i+1]
            for j in range(0, int(leng)):
                num = int(start)+j
                pixels.append(str(num))
                
    
    return pixels

#Given two dataframes, checks the overlap in the run length encoding.
def checkForOne(results, answers):
    print("Checking Answers")
    a1 = pixels(results)
    a2 = pixels(answers)
    
    a1 = set(a1)
    a2 = set(a2)
    
    intersect = a1.intersection(a2)
    d1 = a1.difference(a2)
    d2 = a2.difference(a1)
    print("Percent of target hit")
    print(str(len(intersect) / len(a2)))
    print("Percent of pixels in result that are misses:")
    print(str(len(d1)/len(a1)))
    print("Misses relative to target size")
    print(str(len(d1)/len(a2)))
    
def checkAll(allresults, allanswers):
    for i in range(0, len(allresults)):
        result = allresults[i]
        answer = allanswers[i]
        checkForOne(result, answer)
    

def main():
    res = pandas.read_csv('submission.csv')
    ans = pandas.read_csv('stage1_solution.csv')
    
    

if __name__ == '__main__':
    main()