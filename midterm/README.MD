## 1. DNN8 default, Adam, lr 0.001
0.964 ???

## 2. Retrain with more data
0.967

## 3. Try FocalLoss to handle imbalanced data
No improvement

## 4. Adam -> AdamW
0.97020 !!!

## 5. ReLU -> LeakyReLU
No improvement

## 6. batchsize 128 -> 256
No improvement

## 7. Layers [13] -> [8, 16]
0.97030

## 8. ReLU -> SiLU, Change scheduler hyperparameter
0.9735 !!!

## 9. [8, 16] -> [6, 24]
Accuracy decreases

## 10. lr -> 0.005
0.9728