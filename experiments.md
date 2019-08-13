| Architecture  | Epochs | Batch Size | Learning Rate | Drop Rate | L2 Scale | Train Acc. | Val Acc. | Test Acc. |
| :------------ | :----- | :--------- | :------------ | :-------- | :------- | :--------- | :------- | :-------- |
| LeNet         | 200    | 32         | 0.00001       | 0.5       | 1e-3     | 95.3       | 89.3     | 87.4      |
| Arch. 1       | 200    | 8          | 0.00001       | 0.7       | 1e-3     | 99.8       | 96.3     | 93.7      |
| Arch. 1       | 200    | 32         | 0.00001       | 0.5       | 1e-3     | 99.7       | 94.2     | 92.90     |
| Arch. 2       | 350    | 32         | 0.0001        | 0.5       | 1e-3     | 99.8       | 94.7     | 92.20     |
| Arch. 2       | 250    | 32         | 0.0001        | 0.6       | 1e-3     | 99.4       | 94.0     | 90.50     |
| Arch. 2(Gray) | 300    | 32         | 0.0001        | 0.6       | 1e-3     | 98.8       | 93.2     | 89.60     |
| Arch. 1(Gray) | 200    | 8          | 0.00001       | 0.7       | 1e-3     | 99.7       | 94.9     | 92.4      |
| Arch. 1       | 200    | 8          | 0.00001       | 0.7       | 1e-3     | ---        | ---      | ---       |

## LeNet
With color images
1. Conv Layer 1
   1. Input : 32 x 32 x 3
   2. Filter : 5 x 5 x 3 x 6
   3. Activation : Relu
   4. Output : 28 x 28 x 6
2. Max Pool 1
   1. Input : 28 x 28 x 6
   2. Kernel = 1 x 2 x 2 x 1
   3. Stride = [1, 2, 2, 1]
   4. Padding : Valid
   5. Output : 14 x 14 x 6
3. Conv Layer 2
   1. Input : 14 x 14 x 6
   2. Filter : 5 x 5 x 6 x 16
   3. Activation : Relu
   4. Output : 10 x 10 x 16
4. Max Pool 2
   1. Input : 10 x 10 x 16
   2. Kernel = 1 x 2 x 2 x 1
   3. Stride = [1, 2, 2, 1]
   4. Padding : Valid
   5. Output : 5 x 5 x 16
5. FC 1
   1. Input : 400
   2. Weights : 400 x 120
   3. Biases : 120
   4. Activation : Relu
   5. Output : 120
6. FC 2
   1. Input : 120
   2. Weights : 120 x 84
   3. Biases : 84
   4. Activation : Relu
   5. Output : 84
7. 6. FC 3
   1. Input : 84
   2. Weights : 84 x 43
   3. Biases : 43
   4. Output : 43

## Architecture 1
1. Conv Layer 1
   1. Input : 32 x 32 x 3
   2. Filter : 5 x 5 x 3 x 16
   3. Activation : Relu
   4. Output : 28 x 28 x 16
2. Max Pool 1
   1. Input : 28 x 28 x 16
   2. Kernel = 1 x 2 x 2 x 1
   3. Stride = [1, 2, 2, 1]
   4. Padding : Valid
   5. Output : 14 x 14 x 16
3. Conv Layer 2
   1. Input : 14 x 14 x 16
   2. Filter : 5 x 5 x 16 x 32
   3. Activation : Relu
   4. Output : 10 x 10 x 32
4. Max Pool 2
   1. Input : 10 x 10 x 32
   2. Kernel = 1 x 2 x 2 x 1
   3. Stride = [1, 2, 2, 1]
   4. Padding : Valid
   5. Output : 5 x 5 x 32
5. FC 1
   1. Input : 800
   2. Weights : 800 x 500
   3. Biases : 500
   4. Activation : Relu
   5. Output : 500
6. FC 2
   1. Input : 500
   2. Weights : 500 x 100
   3. Biases : 100
   4. Activation : Relu
   5. Output : 100
7. 6. FC 3
   1. Input : 100
   2. Weights : 100 x 43
   3. Biases : 43
   4. Output : 43


## Architecture 2
A deeper architecture, one extre conv layer. Adapted from LeNet
1. Conv Layer 1
   1. Input : 32 x 32 x 3
   2. Filter : 5 x 5 x 3 x 6
   3. Activation : Relu
   4. Output : 28 x 28 x 6
2. Max Pool 1
   1. Input : 28 x 28 x 6
   2. Kernel = 1 x 2 x 2 x 1
   3. Stride = [1, 2, 2, 1]
   4. Padding : Valid
   5. Output : 14 x 14 x 6
3. Conv Layer 2
   1. Input : 14 x 14 x 6
   2. Filter : 5 x 5 x 6 x 16
   3. Activation : Relu
   4. Output : 10 x 10 x 16
4. Max Pool 2
   1. Input : 10 x 10 x 16
   2. Kernel = 1 x 2 x 2 x 1
   3. Stride = [1, 2, 2, 1]
   4. Padding : Valid
   5. Output : 5 x 5 x 16
5. Conv Layer 3
   1. Input : 5 x 5 x 16
   2. Filter : 3 x 3 x 16 x 32
   3. Activation : Relu
   4. Output : 3 x 3 x 32
6. Max Pool 3
   1. Input : 3 x 3 x 32
   2. Kernel = 1 x 2 x 2 x 1
   3. Stride = [1, 2, 2, 1]
   4. Padding : Valid
   5. Output : 1 x 1 x 32
7. FC 1
   1. Input : 32
   2. Weights : 32 x 120
   3. Biases : 120
   4. Activation : Relu
   5. Output : 120
8. FC 2
   1. Input : 120
   2. Weights : 120 x 84
   3. Biases : 84
   4. Activation : Relu
   5. Output : 84
9. 6. FC 3
   1. Input : 84
   2. Weights : 84 x 43
   3. Biases : 43
   4. Output : 43