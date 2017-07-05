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

function seqDataSet (sWords, wordVecMap) {
  var dataSet = []
  for (let i = 1; i < sWords.length; i++) {
    var sVector = wordVecMap[sWords[i - 1]]
    var tVector = wordVecMap[sWords[i]]
    console.log('Seq: sWord[%d]=%s %j sWord[%d]=%s %j', i - 1, sWords[i - 1], sVector, i, sWords[i], tVector)
    dataSet.push({ input: sVector, output: tVector })
  }
  return dataSet
}

function mtDataSet (sWords, tWords, wordVecMap) {
  var dataSet = []
  for (let i = 0; i < sWords.length; i++) {
    var sVector = wordVecMap[sWords[i]]
    var tVector = wordVecMap[tWords[i]]
    console.log('MT%d: sWord=%s %j tWord=%s %j', i, sWords[i], sVector, tWords[i], tVector)
    dataSet.push({ input: sVector, output: tVector })
  }
  return dataSet
}

function training (dataSet, wordSet) {
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

  return network
}

/*
function genSentence (sWords, wordSet, vecWordMap, network) {
  for (let i = 0; i < sWords.length; i++) {
    var tProb = network.activate(wordVecMap[sWords[i]])
    var max = Math.max.apply(null, tProb)
    var tIndex = tProb.indexOf(max)
    var tVector = Array(wordSet.size).fill(0)
    tVector[tIndex] = 1
    var tWord = vecWordMap[tVector] // Object.keys(v).find(key => Vector[key].toString() === zeros.toString());
    console.log('sWord=%s gen:tWord=%s tProb=%j', sWords[i], tWord, tProb)
  }
}
*/

function sentenceMt (sWords, wordSet, vecWordMap, network) {
  console.log('======== source ===========')
  for (let i = 0; i < sWords.length; i++) {
    var sWord = sWords[i]
    console.log('source:word=%s', sWord)
    let prob = network.activate(wordVecMap[sWords[i]])
    let word = prob2word(prob, wordSet, vecWordMap)
/*    
    var max = Math.max.apply(null, tProb)
    var tIndex = tProb.indexOf(max)
    var tVector = Array(wordSet.size).fill(0)
    tVector[tIndex] = 1
    var tWord = vecWordMap[tVector] // Object.keys(v).find(key => Vector[key].toString() === zeros.toString());
*/    
  }
  console.log('======== mt ===========')
  var lastWord = '='
  for (let i = 0; i < sWords.length; i++) {
    let prob = network.activate(wordVecMap[lastWord])
    let word = prob2word(prob, wordSet, vecWordMap)
    console.log('target:word=%s', word)
    lastWord = word
  }
  network.activate(wordVecMap['.'])
}

function prob2word (tProb, wordSet, vecWordMap) {
  var max = Math.max.apply(null, tProb)
  var tIndex = tProb.indexOf(max)
  var tVector = Array(wordSet.size).fill(0)
  tVector[tIndex] = 1
  var tWord = vecWordMap[tVector] // Object.keys(v).find(key => Vector[key].toString() === zeros.toString());  
  return tWord
}
/*
function mtSentence (sWords, wordSet, vecWordMap, seqNet, mtNet) {
  for (let i = 0; i < sWords.length; i++) {
    var seqProb = seqNet.activate(wordVecMap[sWords[i]])
    var nextWord = prob2word(seqProb)
    console.log('sWord=%s nextWord=%s tProb=%j', sWords[i], nextWord, seqProb)
    var mtProb = mtNet.activate(wordVecMap[sWords[i]])
    var mtWord = prob2word(mtProb)
    console.log('sWord=%s mtWord=%s tProb=%j', sWords[i], mtWord, mtProb)
  }
}
*/
// var seqText = '小 狗 = dog little . 黃 狗 = dog yellow . 小 黃 狗 = dog yellow little . 小 貓 = cat little . 黑 貓 = cat black . 小 黑 貓 = cat black little .'
// var seqText = '. 黃 狗 = dog yellow . 小 狗 = dog little . 小 貓 = cat little . 小 黑 貓 = cat black little .'
var seqText = '黑 狗 = dog black . 小 狗 = dog little . 小 黑 狗 = dog black little .'
seqText = seqText + seqText + seqText
var mtSource = ''
var mtTarget = ''
// var mtSource = '黃     狗  小     黑     貓  .'
// var mtTarget = 'yellow dog little black cat .'

var seqWords = text2words(seqText)
var mtWords = text2words(mtTarget + ' ' + mtSource)
var wordSet = words2set(seqWords.concat(mtWords))
var words = Array.from(wordSet)
console.log('words = %j', words)
var {wordVecMap, vecWordMap} = oneHotMap(words)

/*
var sourceWords = text2words(mtSource)
var targetWords = text2words(mtTarget)
console.log('sourceWords=%j', sourceWords)
var mtData = mtDataSet(sourceWords, targetWords, wordVecMap)
var mtNet = training(mtData, wordSet)
*/

var seqData = seqDataSet(seqWords, wordVecMap)
var seqNet = training(seqData, wordSet)

var sLines = '狗.黑 狗.小 狗.小 黑 狗.小 小 狗.黑 小 狗'.split('.')
console.log('sLines=%j', sLines)
for (let line of sLines) {
  let sWords = text2words(line)
  console.log('sWords = %j', sWords)
  sentenceMt(sWords, wordSet, vecWordMap, seqNet)
}

// mtSentence(testWords, wordSet, vecWordMap, seqNet, mtNet)
