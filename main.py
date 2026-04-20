
import numpy as np
import matplotlib.pyplot as plt
# plt.imshow(matrix, cmap='gray')
# plt.show()
import csv
zeroOneData = []
results = []

def relu(x):
    return np.maximum(0, x)
def reluAbleitung(x):
    if x < 0:
        return 0
    else:
        return 1 
def reluAbleitungVektor(x):
    return (x > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
with open('mnist_train.csv', mode ='r',)as file:
    trainData = csv.reader(file)
    for lines in trainData:
        number = int(lines[0])
        # if number == 0 or number == 1:

        results.append(number)
        lines.pop(0)
        matrix = np.array(lines).astype(int) / 255
        # matrix = matrix.reshape(28,28)
        
        
        zeroOneData.append(matrix)
         

testData = []
testResults = []           
with open('mnist_test.csv', mode ='r',)as file:
    trainData = csv.reader(file)
    for lines in trainData:
        number = int(lines[0])
        # if number == 0 or number == 1:
        testResults.append(number)
        lines.pop(0)
        matrix = np.array(lines).astype(int)/255
        # matrix = matrix.reshape(28,28)
        testData.append(matrix)

class neuronalNetwork:
    def __init__(self,inputs,neurons,outputs,hiddenLayers,learningRate = 0.1):
        self.weights = []
        self.bias = []
        self.learningRate = learningRate
        self.weights.append(np.random.randn(inputs,neurons) * 0.1)
        self.bias.append(np.zeros((1,neurons)))
        for i in range(hiddenLayers):
            self.weights.append(np.random.randn(neurons,neurons) * 0.1)
            self.bias.append(np.zeros((1,neurons)))
        self.weights.append(np.random.randn(neurons,outputs) * 0.1)
        self.bias.append(np.zeros((1,outputs)))
    def forward(self,x,l=0):
        self.a = []
        self.a = [np.array(x).reshape(1,784)]
        self.z = []
        # Später zwischen normalen Layern und Hidden Layer unterscheiden
        for i in range(len(self.weights)): 
            
            z = np.dot(self.a[-1],self.weights[i]) + self.bias[i]
            self.z.append(z)
            if i == len(self.weights)-1:
                a =softmax(z)
                print(a)
            else:
                a = relu(z)
            self.a.append(a)

    def backward(self,result):
        ownResult = self.a[-1][0]
        error = np.zeros((1, 10))
        error[0,result] = 1
        print("Sollte: " + str(result))
        print("Ist: " + str(ownResult))

        # for i in range(len(ownResult)):
        #     if i == result:
        #         error.append((ownResult[i] - 1) )

        #     else:
        #         error.append(ownResult[i]  )
                # * reluAbleitung(ownResult[i])
        
        # errorNeu = np.array(error).reshape(1, -1)
        errorNeu = ownResult - error
        print("OutpuTError: " + str(errorNeu))
        for n in reversed(range(len(self.weights))):
            error = errorNeu

            if(n >0):
                if(n > 1):
                    errorNeu = np.dot(error,self.weights[n].T) * reluAbleitungVektor(self.z[n-1])
                # * reluAbleitung(self.z[n-1])
                else:
                    errorNeu = np.dot(error,self.weights[n].T)
            self.weights[n] = self.weights[n] - self.learningRate * np.dot(self.a[n].T,error)
            self.bias[n] = self.bias[n] - self.learningRate * error  

brain = neuronalNetwork(784,128,10,3,0.01)       
for i, matrix in enumerate(zeroOneData):
   brain.forward(matrix)
    
   brain.backward(results[i])

print("Lernen fertig.............................................................................")
fehler = 0
for i,matrix in enumerate(testData):
    brain.forward(matrix)
    print("Sollte: " +str(testResults[i])  )
    output= brain.a[-1][0].flatten()

    maxZahl = np.argmax(output)
    print("Erkannt: " + str(maxZahl) + " zu " + str(np.round(output[maxZahl] *100,2)) + "%")
    # print(np.round(output * 100, 2))
    if testResults[i] != maxZahl:
        fehler+=1
        print("Falsch....")
quitient =  fehler / len(testData) * 100
print(fehler)
print(len(testData))
print("Fehler in Prozent: "+ str(quitient))

