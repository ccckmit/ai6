// http://cs.stanford.edu/people/karpathy/convnetjs/ (有 web)
// -- https://www.npmjs.com/package/convnetjs-ts
// https://github.com/junku901/dnn (已吸收)
// https://github.com/cazala/mnist
// article : https://blog.webkid.io/neural-networks-in-javascript/
// npm install synaptic --save(讚! 吸收中)
// -- 記得要看 wiki (https://github.com/cazala/synaptic/wiki)
// -- https://blog.webkid.io/neural-networks-in-javascript/ 有 NMIST 手寫辨識範例
// https://wagenaartje.github.io/neataptic/ (讚！待吸收, Flexible neural network library with advanced neuroevolution)
// -- https://wagenaartje.github.io/neataptic/docs/tutorials/tutorials/
// Node tensorflow -- https://github.com/peterbraden/node-tensorflow (還不算完整)
//  var NN = ai6.NN = module.exports = require('synaptic')
// 書籍： https://page.mi.fu-berlin.de/rojas/neural/
module.exports = function (ai6) {
  var NN = ai6.NN = {}
  var T = ai6.j6.T

  // ref: https://wagenaartje.github.io/neataptic/docs/methods/activation/
  NN.sigmoid = function (x) {
    var sigmoid = (1.0 / (1 + Math.exp(-x)))
    if (sigmoid === 1) {
      //   console.warn('Something Wrong!! Sigmoid Function returns 1. Probably javascript float precision problem?\nSlightly Controlled value to 1 - 1e-14')
      sigmoid = 0.99999999999999 // Javascript Float Precision Problem.. This is a limit of javascript.
    } else if (sigmoid === 0) {
      //  console.warn('Something Wrong!! Sigmoid Function returns 0. Probably javascript float precision problem?\nSlightly Controlled value to 1e-14')
      sigmoid = 1e-14
    }
    return sigmoid // sigmoid cannot be 0 or 1;
  }

  NN.dSigmoid = function (x) {
    var a = NN.sigmoid(x)
    return a * (1.0 - a)
  }

  NN.logistic = NN.sigmoid

  NN.tanh = Math.tanh

  NN.relu = (x) => (x >= 0) ? x : 0

  NN.step = (x) => (x >= 0) ? 1 : 0

  NN.bipolar = (x) => (x >= 0) ? 1 : -1

  NN.gaussian = (x) => Math.exp(-x * x)

/*
  NN.softmaxVec = function (vec) {
    var max = vec.max()
    var preSoftmaxVec = vec.map((x) => Math.exp(x - max))
    var preSoftmaxSum = preSoftmaxVec.sum()
    return preSoftmaxVec.map((x) => (x / preSoftmaxSum))
  }

  NN.softmaxMat = function (m) {
    var len = m.length
    var r = []
    for (var i = 0; i < len; i++) {
      r[i] = NN.softmaxVec(m[i])
    }
    return r
  }
*/
  NN.binarySample = function (m) {
    return m.map1((x) => (Math.random() < x) ? 1 : 0)
  }

  NN.binaryCrossEntropy = function (x, y) {
    var a = T.map2(x, y, function (px, py) {
      return px * Math.log(py)
    })
    var b = T.map2(x, y, function (px, py) {
      return (1 - px) * Math.log(1 - py)
    })
    var entropy = -a.madd(b).colSum().mean() // -(a+b).colMean()
    return entropy
  }

  NN.feedForward = function (layers, x) {
    var inputLayers = []
    inputLayers.push(x)
    for (let i = 0; i < layers.length; i++) {
      inputLayers.push(layers[i].output(inputLayers[i]))
    }
    return { inputLayers: inputLayers, output: inputLayers[inputLayers.length - 1] }
  }

  NN.backPropagate = function (y, output, layers, inputLayers, dActivation) {
    var n = layers.length - 1
    var delta = new Array(n)
    var linearOutput = layers[n].linearOutput(inputLayers[n]).map1(dActivation)
    delta[n] = y.msub(output).mmul(linearOutput)
    for (let i = n - 1; i >= 0; i--) {
      var o = layers[i].linearOutput(inputLayers[i]).map1(dActivation)
      delta[i] = layers[i + 1].backPropagate(delta[i + 1]).mmul(o)
    }
    return delta
  }

  NN.updateWeights = function (x, layers, inputLayers, delta) {
    for (let i = 0; i < layers.length; i++) {
      var xlen = x.length
      var deltaW = inputLayers[i].tr().mdot(delta[i]).map1((x) => x / xlen)
      var deltaB = delta[i].colMean()
      layers[i].W = layers[i].W.madd(deltaW)
      layers[i].b = layers[i].b.vadd(deltaB)
    }
  }

  // ref : https://github.com/wagenaartje/neataptic/blob/master/src/layer.js
  var neataptic = require('neataptic')
  NN.HiddenLayer = require('./hiddenLayer')(ai6)
  NN.RBM = require('./rbm')(ai6)
  NN.MLP = require('./mlp')(ai6)
  NN.DBN = require('./dbn')(ai6)
  NN.CRBM = require('./crbm')(ai6)
  NN.CDBN = require('./cdbn')(ai6)
  NN.Perceptron = neataptic.Architect.Perceptron
  NN.LSTM = neataptic.Architect.LSTM
  NN.NARX = neataptic.Architect.NARX
  NN.GRU = neataptic.Architect.GRU
  NN.Random = neataptic.Architect.Random
  NN.Hopfield = neataptic.Architect.Hopfield
  NN.Memory = neataptic.Architect.Memory
  NN.Dense = neataptic.Architect.Dense
  return NN
}
