module.exports = function (ai6) {
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
  var NN = ai6.NN = {}
/*
  NN.Perceptron = NN.Architect.Perceptron
  NN.LSTM = NN.Architect.LSTM
  NN.Liquid = NN.Architect.Liquid
  NN.Hopfield = NN.Architect.Hopfield
*/
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

  NN.binarySample = function (m) {
    return m.map1((x) => (Math.random() < x) ? 1 : 0)
  }

  NN.HiddenLayer = require('./hiddenLayer')(ai6)
  NN.RBM = require('./rbm')(ai6)
  NN.MLP = require('./mlp')(ai6)
  NN.DBN = require('./dbn')(ai6)
  NN.CRBM = require('./crbm')(ai6)
  NN.CDBN = require('./cdbn')(ai6)
  return NN
}
