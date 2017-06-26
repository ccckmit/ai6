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
  var Opt = ai6.Optimize = {}

  Opt.Solution = require('./solution')(ai6)
  Opt.hillClimbing = require('./hillClimbing')
  Opt.simulatedAnnealing = require('./simulatedAnnealing')
  return Opt
}
