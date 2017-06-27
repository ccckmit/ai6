module.exports = function (ai6) {
  var j6 = ai6.j6
  var M = j6.M
  var V = j6.V
  var NN = ai6.NN

  var HiddenLayer = function (settings) {
    this.input = settings['input']
    var a = 1.0 / settings['nIn']
    this.W = settings['W'] || M.random(settings['nIn'], settings['nOut'], -a, a)
    this.b = settings['b'] || V.new(settings['nOut'])
    this.activation = settings['activation'] || NN.sigmoid
  }

  HiddenLayer.prototype.output = function (input) {
    this.input = input || this.input
    return this.linearOutput(this.input).map1(this.activation)
  }

  HiddenLayer.prototype.linearOutput = function (input) { // before activation
    this.input = input || this.input
    var linearOutput = M.addv(M.dot(this.input, this.W), this.b)
    return linearOutput
  }

  HiddenLayer.prototype.backPropagate = function (input) { // example+num * nOut matrix
    if (typeof input === 'undefined') throw new Error('No BackPropagation Input.')
    var linearOutput = input.mdot(this.W.tr())
    return linearOutput
  }

  HiddenLayer.prototype.sampleHgivenV = function (input) {
    this.input = input || this.input
    var hMean = this.output()
    var hSample = ai6.NN.binarySample(hMean) // Gibb's random sampling
    return hSample
  }

  return HiddenLayer
}
