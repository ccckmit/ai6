/* eslint-disable camelcase */
module.exports = function (ai6) {
  var NN = ai6.NN
  var j6 = ai6.j6
  var T = j6.T

  var MLP = function (settings) {
    this.x = settings['input']
    this.y = settings['label']
    this.sigmoidLayers = []
    this.nLayers = settings['hiddenLayerSizes'].length
    for (var i = 0; i < this.nLayers + 1; i++) {
      var inputSize = (i === 0) ? settings['nIns'] : settings['hiddenLayerSizes'][i - 1]
      var layerInput = (i === 0) ? this.x : this.sigmoidLayers[this.sigmoidLayers.length - 1].sampleHgivenV()
      var nOut = (i === this.nLayers) ? settings['nOuts'] : settings['hiddenLayerSizes'][i]
      var sigmoidLayer = new ai6.NN.HiddenLayer({
        input: layerInput,
        nIn: inputSize,
        nOut: nOut,
        activation: ai6.NN.sigmoid,
        W: (typeof settings['w_array'] === 'undefined') ? undefined : settings['w_array'][i],
        b: (typeof settings['b_array'] === 'undefined') ? undefined : settings['b_array'][i]
      })
      this.sigmoidLayers.push(sigmoidLayer)
    }
  }

  MLP.prototype.train = function (settings) {
  //  var lr = settings['lr'] || 0.6
    var epochs = settings['epochs'] || 1000
    var currentProgress = 1
    for (var epoch = 0; epoch < epochs; epoch++) {
      // Feed Forward
      var i
      var layerInput = []
      layerInput.push(this.x)
      for (i = 0; i < this.nLayers + 1; i++) {
        layerInput.push(this.sigmoidLayers[i].output(layerInput[i]))
      }
      var output = layerInput[this.nLayers + 1]
      // Back Propagation
      var delta = new Array(this.nLayers + 1)
      var linearOutput = this.sigmoidLayers[this.nLayers].linearOutput(layerInput[this.nLayers]).map1(NN.dSigmoid)
      delta[this.nLayers] = this.y.msub(output).mmul(linearOutput)
      for (i = this.nLayers - 1; i >= 0; i--) {
        var o = this.sigmoidLayers[i].linearOutput(layerInput[i]).map1(NN.dSigmoid)
        delta[i] = this.sigmoidLayers[i + 1].backPropagate(delta[i + 1]).mmul(o)
      }
      // Update Weight, Bias
      for (i = 0; i < this.nLayers + 1; i++) {
        var xlen = this.x.length
        var deltaW = layerInput[i].tr().mdot(delta[i]).map1((x) => x / xlen)
        var deltaB = delta[i].colMean()
        this.sigmoidLayers[i].W = this.sigmoidLayers[i].W.madd(deltaW)
        this.sigmoidLayers[i].b = this.sigmoidLayers[i].b.vadd(deltaB)
      }
      var progress = (1.0 * (epoch / epochs)) * 100
      if (progress > currentProgress) {
        console.log('MLP', progress.toFixed(0), '% Completed.')
        currentProgress += 8
      }
    }
    console.log('MLP Final Cross Entropy : ', this.getReconstructionCrossEntropy())
  }

  MLP.prototype.getReconstructionCrossEntropy = function () {
    var reconstructedOutput = this.predict(this.x)
    var a = T.map2(this.y, reconstructedOutput, function (x, y) {
      return x * Math.log(y)
    })
    var b = T.map2(this.y, reconstructedOutput, function (x, y) {
      return (1 - x) * Math.log(1 - y)
    })
    var crossEntropy = -a.madd(b).colSum().mean() // -(a+b).colMean()
    return crossEntropy
  }

  MLP.prototype.predict = function (x) {
    var output = x
    for (var i = 0; i < this.nLayers + 1; i++) {
      output = this.sigmoidLayers[i].output(output)
    }
    return output
  }

  return MLP
}
