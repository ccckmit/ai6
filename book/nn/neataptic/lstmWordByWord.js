// ref: https://jsfiddle.net/wagenaartje/k23zbf0f/1/
var neataptic = require('neataptic')
var Methods = neataptic.Methods
var Config = neataptic.Config
var Architect = neataptic.Architect

Config.warnings = false

function text2words (text) {
  return text.toLowerCase().split(/\s+/)
}

function words2set (words) {
  var wordSet = new Set()
  for (let word of words) {
    wordSet.add(word)
  }
  return wordSet
}

Set.prototype.union = function (setB) {
  var union = new Set(this)
  for (var elem of setB) {
    union.add(elem)
  }
  return union
}

/** One-hot encode them */
function oneHotMap (words) {
  let wordVecMap = {}
  let vecWordMap = {}
  for (let i = 0; i < words.length; i++) {
    let word = words[i]
    let vector = Array(words.length).fill(0)
    vector[i] = 1
    wordVecMap[word] = vector
    vecWordMap[vector] = word
  }
  return {wordVecMap: wordVecMap, vecWordMap: vecWordMap}
}

/** Prepare the data-set */
function buildDataSet (sWords, wordSet) {
  var dataSet = []
  for (let i = 1; i < sWords.length; i++) {
    var sVector = wordVecMap[sWords[i - 1]]
    var tVector = wordVecMap[sWords[i]]
    console.log('sWord[%d]=%s %j sWord[%d]=%s %j', i - 1, sWords[i - 1], sVector, i, sWords[i], tVector)
    dataSet.push({ input: sVector, output: tVector })
  }
  return dataSet
}

function genSentence (sWords, wordSet, vecWordMap, network) {
  for (let i = 0; i < sWords.length; i++) {
    var tProb = network.activate(wordVecMap[sWords[i]])
    var max = Math.max.apply(null, tProb)
    var tIndex = tProb.indexOf(max)
    var tVector = Array(wordSet.size).fill(0)
    tVector[tIndex] = 1
    var tWord = vecWordMap[tVector] // Object.keys(v).find(key => Vector[key].toString() === zeros.toString());
    console.log('sWord=%s mt:tWord=%s tProb=%j', sWords[i], tWord, tProb)
  }
}

// var sText = 'no free lunch . he is an free worker'
var sText = '小 狗 . 黃 狗 . 小 黃 狗 . 小 貓 . 黑 貓 . 小 黑 貓 .'
var sWords = text2words(sText)
console.log('sWords=%j', sWords)
var sWordSet = words2set(sWords)
var wordSet = sWordSet
var words = Array.from(wordSet)
var {wordVecMap, vecWordMap} = oneHotMap(words)

var dataSet = buildDataSet(sWords, wordSet)
console.log('dataSet=%j', dataSet)
var network = new Architect.LSTM(wordSet.size, 10, wordSet.size)

console.log('Network conns', network.connections.length, 'nodes', network.nodes.length)

network.train(dataSet, {
  log: 1,
  rate: 0.1,
  cost: Methods.Cost.MSE,
  error: 0.005,
  clear: true
})

var sText2 = '黃 狗 . 小 黃 . 小 . 黑 . 小 黑 .'
var sWords2 = text2words(sText2)
genSentence(sWords2, wordSet, vecWordMap, network)
