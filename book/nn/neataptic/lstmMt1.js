// ref: https://jsfiddle.net/wagenaartje/k23zbf0f/1/
var neataptic = require('neataptic')
// var Neat = neataptic.Neat
var Methods = neataptic.Methods
var Config = neataptic.Config
var Architect = neataptic.Architect
// var Network = neataptic.Network
// var Node = neataptic.Node

/** Turn off warnings */
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
function mtDataSet (sWords, tWords, wordSet) {
  var dataSet = []
  for (let i = 0; i < sWords.length; i++) {
    var sVector = wordVecMap[sWords[i]]
    var tVector = wordVecMap[tWords[i]]
    console.log('sWord=%s %j tWord=%s %j', sWords[i], sVector, tWords[i], tVector)
    dataSet.push({ input: sVector, output: tVector })
  }
  return dataSet
}

function mt (sWords, wordSet, vecWordMap, network) {
  for (let i = 0; i < sWords.length; i++) {
    var tProb = network.activate(wordVecMap[sWords[i]])
    var max = Math.max.apply(null, tProb)
    var tIndex = tProb.indexOf(max)
    var tVector = Array(wordSet.size).fill(0)
    tVector[tIndex] = 1
    // 寫到這裡，下面繼續 (找出 tVector 對應的 tWord 然後印出)
    var tWord = vecWordMap[tVector] // Object.keys(v).find(key => Vector[key].toString() === zeros.toString());
    console.log('sWord=%s mt:tWord=%s', sWords[i], tWord)
//    output = network.activate(zeros)
  }
}

// var sText = `man eat fish . people love cat . cat eat fish .`
// var tText = `人  吃  魚    . 人  愛   貓  . 貓  吃  魚    .`
// var sText = `river bank . work in a bank .`
// var tText = `河 岸 . 工作 在 一家 銀行 .`
var sText = 'no free lunch . he is an free worker'
var tText = '沒有 免費的 午餐 . 他 是 一個 自由 工作者'
var sWords = text2words(sText); var tWords = text2words(tText)
console.log('sWords=%j', sWords)
console.log('tWords=%j', tWords)
var sWordSet = words2set(sWords); var tWordSet = words2set(tWords)
// console.log('sWordSet=%j sWordSet.size=%d',  Array.from(sWordSet), sWordSet.size)
var wordSet = sWordSet.union(tWordSet)
var words = Array.from(wordSet)
var {wordVecMap, vecWordMap} = oneHotMap(words)
// console.log('wordVecMap=%s size=%d', JSON.stringify(wordVecMap), wordVecMap.size)
var dataSet = mtDataSet(sWords, tWords, wordSet)
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

mt(sWords, wordSet, vecWordMap, network)

// var sWords2 = text2words('river bank . work in bank . river bank . river bank .')
// mt(sWords2, wordSet, vecWordMap, network)
