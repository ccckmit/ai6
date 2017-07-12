# ADNN -- 包含自動微分的神經網路套件

https://github.com/dritchie/adnn

Javascript neural networks on top of general scalar/tensor reverse-mode automatic differentiation.

## Example 1 : nnXor.js

file: [nnXor.js](nnXor.js)

run: 

```
D:\Dropbox\ai6\book\nn\adnn>node nnXor.js
buildNet
loadData
nnTrain
predict
input={"dims":[2],"length":2,"data":{"0":0,"1":0}} probs={"dims":[2],"length":2,
"data":{"0":0.9995266414085302,"1":0.0004733585914697014}}
input={"dims":[2],"length":2,"data":{"0":0,"1":1}} probs={"dims":[2],"length":2,
"data":{"0":0.00277378478556229,"1":0.9972262152144377}}
input={"dims":[2],"length":2,"data":{"0":1,"1":0}} probs={"dims":[2],"length":2,
"data":{"0":0.002685249616197764,"1":0.9973147503838022}}
input={"dims":[2],"length":2,"data":{"0":1,"1":1}} probs={"dims":[2],"length":2,
"data":{"0":0.9950174687367755,"1":0.004982531263224475}}
```
