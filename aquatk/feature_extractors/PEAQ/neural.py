import numpy as np

# Define constants and weights
amin = np.array([393.916656, 361.965332, -24.045116, 1.110661, -0.206623,
                 0.074318, 1.113683, 0.950345, 0.029985, 0.000101, 0.0])
amax = np.array([921.0, 881.131226, 16.212030, 107.137772, 2.886017,
                 13.933351, 63.257874, 1145.018555, 14.819740, 1.0, 1.0])
wx = np.array([[-0.502657, 0.436333, 1.219602],
               [4.307481, 3.246017, 1.123743],
               [4.984241, -2.211189, -0.192096],
               [0.051056, -1.762424, 4.331315],
               [2.321580, 1.789971, -0.754560],
               [-5.303901, -3.452257, -10.814982],
               [2.730991, -6.111805, 1.519223],
               [0.624950, -1.331523, -5.955151],
               [3.102889, 0.871260, -5.922878],
               [-1.051468, -0.939882, -0.142913],
               [-1.804679, -0.503610, -0.620456]])
wy = np.array([-3.817048, 4.107138, 4.629582, -0.307594])
bmin = -3.98
bmax = 0.22
I = 11  # Number of input neurons
J = 3  # Number of hidden neurons

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def neural(processed):
    x = np.array([processed["BandwidthRefb"], processed["BandwidthTestb"], processed["TotalNMRb"],
                  processed["WinModDiff1b"], processed["ADBb"], processed["EHSb"], processed["AvgModDiff1b"],
                  processed["AvgModDiff2b"], processed["RmsNoiseLoudb"], processed["MFPDb"], processed["RelDistFramesb"]])
    
    
    x_norm = (x - amin) / (amax - amin)
    

    sum1 = np.dot(x_norm, wx)  
    hidden_output = sigmoid(sum1)
    
    
    sum2 = np.dot(wy[:-1], hidden_output)  
    DI = wy[-1] + sum2  
    ODG = bmin + (bmax - bmin) * sigmoid(DI)
    
    return {"DI": DI, "ODG": ODG}





