# 簡易的深度學習機器翻譯系統

搞了好幾天，終於弄出一版《基於 LSTM 深度學習的機器翻譯系統》。

有了這次經驗，終於可以體會為何深度學習的機器翻譯為何感覺那麼的順暢了。

順暢的主要原因是： 透過 LSTM 去講故事，而不是一個字一個字的對翻！

LSTM 本來就是用來做《序列學習》的，所以講出來的東西通常都很順暢，只要在選字上面選對了，之後用 LSTM 產生出來就會是《很順暢的翻譯》了 ...

## 安裝方法

記得檢查你的 node.js 是否比 7.6 版還新，一定要大於 7.6 版，否則無法成功執行！

```
$ git clone https://github.com/ccckmit/ai6.git
$ cd ai6
$ npm install
$ cd book/mt/
```

## 執行方法

訓練階段

```
$ node mtTrain data/mtDogCat.txt data/mtDogCat.json

Lstm.setting.words=["[#start#]","一","隻","狗","=","a","dog","↓","貓","cat","小","pupp
y","kitten","黑","black","[#end#]"]
Lstm.setting.w2i={"[#start#]":0,"一":1,"隻":2,"狗":3,"=":4,"a":5,"dog":6,"↓":7,"貓":8,
"cat":9,"小":10,"puppy":11,"kitten":12,"黑":13,"black":14,"[#end#]":15}
Network conns 4608 nodes 176
iteration 1 error 0.14293759569898304 rate 0.1
iteration 2 error 0.07522323385036442 rate 0.1
iteration 3 error 0.06458491480240852 rate 0.1
iteration 4 error 0.057082455586521925 rate 0.1
...

iteration 29 error 0.011055015583765218 rate 0.1
iteration 30 error 0.010528333640657356 rate 0.1
iteration 31 error 0.010034839611344567 rate 0.1
iteration 32 error 0.009572102658143247 rate 0.1
Network conns 17408 nodes 336
iteration 1 error 0.07548083477579114 rate 0.1
iteration 2 error 0.05634446473695932 rate 0.1
iteration 3 error 0.04999356582740921 rate 0.1
iteration 4 error 0.04600692322456007 rate 0.1
iteration 5 error 0.04324447514472449 rate 0.1
iteration 6 error 0.04115684679871102 rate 0.1
iteration 7 error 0.03948834398834224 rate 0.1
...
iteration 185 error 0.011420131758988615 rate 0.1
iteration 186 error 0.011057974053230687 rate 0.1
iteration 187 error 0.010670960617216422 rate 0.1
iteration 188 error 0.01030410947451134 rate 0.1
iteration 189 error 0.009888661334289672 rate 0.1
```

翻譯測試

```
$node mtPredict data/mtDogCat.json data/mtDogCat.tst

===== predict:黑 狗 ====
(candidates)
6:word=dog p=0.6293747319590483
11:word=puppy p=0.4925818381156842
14:word=black p=0.9735647058034166
黑 狗 => ["black","dog"]
===== predict:黑 貓 ====
(candidates)
9:word=cat p=0.6277232294419235
12:word=kitten p=0.4100198691268047
14:word=black p=0.9718996045247326
黑 貓 => ["black","cat"]
===== predict:小 黑 狗 ====
(candidates)
11:word=puppy p=0.899685281474142
14:word=black p=0.9593225891303813
小 黑 狗 => ["black","puppy"]
===== predict:小 黑 貓 ====
(candidates)
12:word=kitten p=0.9029576629614735
14:word=black p=0.9581542019122061
小 黑 貓 => ["black","kitten"]
===== predict:一 隻 小 黑 狗 ====
(candidates)
5:word=a p=0.9465430746104574
11:word=puppy p=0.8404989865335679
14:word=black p=0.8370177721121996
一 隻 小 黑 狗 => ["a","black","puppy"]
===== predict:一 隻 小 黑 貓 ====
(candidates)
5:word=a p=0.9467351908196237
12:word=kitten p=0.8632927435477813
14:word=black p=0.8332358050625766
一 隻 小 黑 貓 => ["a","black","kitten"]
===== predict:一 隻 小 狗 ====
(candidates)
5:word=a p=0.9538437108394167
11:word=puppy p=0.8218844895026884
一 隻 小 狗 => ["a","puppy"]
===== predict:一 隻 小 貓 ====
(candidates)
5:word=a p=0.9543191669903888
12:word=kitten p=0.8440008412669697
一 隻 小 貓 => ["a","kitten"]
===== predict:一 隻 狗 ====
(candidates)
5:word=a p=0.9506442005497401
6:word=dog p=0.566020170583094
11:word=puppy p=0.255906735388155
一 隻 狗 => ["a","dog"]
===== predict:一 隻 貓 ====
(candidates)
5:word=a p=0.9512096209928514
9:word=cat p=0.6540679175612915
12:word=kitten p=0.2571352729084012
一 隻 貓 => ["a","cat"]
===== predict:小 狗 ====
(candidates)
11:word=puppy p=0.8807980214373727
14:word=black p=0.24660209142124473
小 狗 => ["puppy"]
===== predict:小 貓 ====
(candidates)
12:word=kitten p=0.8860884111304668
14:word=black p=0.2415756184834393
小 貓 => ["kitten"]
```
