// ref: https://jsfiddle.net/wagenaartje/k23zbf0f/1/
var neataptic = require('neataptic')
var W = require('./words')

var LSTM = module.exports = {
  mode: 'char',
  startWord: '[#start#]'
}

LSTM.init = function () {
  neataptic.Config.warnings = false
}

LSTM.seqDataSet = function (sWords, wordVecMap) {
  var dataSet = []
  for (let i = 1; i < sWords.length; i++) {
    var sVector = wordVecMap[sWords[i - 1]]
    var tVector = wordVecMap[sWords[i]]
    dataSet.push({ input: sVector, output: tVector })
  }
  return dataSet
}

LSTM.dataTrain = function (dataSet, words) {
  var network = new neataptic.Architect.LSTM(words.length, Math.min(words.length, 100), words.length)

  console.log('Network conns', network.connections.length, 'nodes', network.nodes.length)

  network.train(dataSet, {
    log: 1,
    rate: 0.1,
    cost: neataptic.Methods.Cost.MSE,
    error: 0.01,
    clear: true
  })

  return network
}

LSTM.seqTrain = function (seqText) {
  console.log('seqText=%j', seqText)
  var seqWords = (LSTM.mode === 'char') ? W.text2chars(seqText) : W.text2words(seqText)
  seqWords = [LSTM.startWord].concat(seqWords)
  var wordSet = W.words2set(seqWords)
  LSTM.words = Array.from(wordSet)
  console.log('words = %j', LSTM.words)
  var {wordVecMap, vecWordMap} = W.oneHotMap(LSTM.words)
  LSTM.wordVecMap = wordVecMap
  LSTM.vecWordMap = vecWordMap

  var seqData = LSTM.seqDataSet(seqWords, LSTM.wordVecMap)
  LSTM.network = LSTM.dataTrain(seqData, LSTM.words)
}

LSTM.genSentence = function (sWords, stops = [], maxLen = 10) {
  var genList = []
  var lastWord = LSTM.startWord
  var word
  for (let i = 0; i < sWords.length; i++) {
    genList.push(sWords[i])
    let prob = LSTM.network.activate(LSTM.wordVecMap[sWords[i]])
    word = W.prob2word(prob, LSTM.words, LSTM.vecWordMap)
    lastWord = word
  }
  for (let i = 0; i < maxLen; i++) {
    genList.push(word)
    let prob = LSTM.network.activate(LSTM.wordVecMap[lastWord])
    word = W.prob2word(prob, LSTM.words, LSTM.vecWordMap)
    if (stops.indexOf(word) >= 0) break
    lastWord = word
  }
  genList.push(word)
  return genList
}

LSTM.genLines = function (sLines, stops = [], maxLen = 100, postTriggers = [], preTrigger = []) {
  for (let line of sLines) {
    let prefix = preTrigger.concat(W.text2words(line)).concat(postTriggers)
    var genList = LSTM.genSentence(prefix, stops, maxLen)
    var genText = (LSTM.mode === 'char') ? genList.join('') : genList.join(' ')
    console.log('======== gen (prefix=%j) ===========', prefix)
    console.log('%s', genText)
  }
}

LSTM.toJSON = function () {
  var obj = {
    mode: LSTM.mode,
    words: LSTM.words,
    wordVecMap: LSTM.wordVecMap,
    vecWordMap: LSTM.vecWordMap,
    network: LSTM.network.toJSON()
  }
  var json = JSON.stringify(obj, null, 2)
  return json
}

LSTM.fromJSON = function (json) {
  var obj = JSON.parse(json)
  LSTM.mode = obj.mode
  LSTM.words = obj.words
  LSTM.wordVecMap = obj.wordVecMap
  LSTM.vecWordMap = obj.vecWordMap
  LSTM.network = neataptic.Network.fromJSON(obj.network)
}
