var W = require('./words')
var LMT = module.exports = require('./lstm')

LMT.text2vector = function (text) {
  let words = W.text2words(text)
  let vector = W.words2vector(words, LMT.words.length, LMT.w2i)
  return vector
}

LMT.s2tData = function (sLine, tLine) {
  return { input: LMT.text2vector(sLine), output: LMT.text2vector(tLine) }
}

LMT.mtDataSet = function (lines) {
//  console.log('lines=%j', lines)
  var dataSet = []
  for (let i = 0; i < lines.length; i++) {
    var parts = lines[i].split('=')
    if (parts.length < 2) continue
    var [sLine, tLine] = parts
//    console.log('s=%s t=%s', sLine, tLine)
    var s2tData = LMT.s2tData(sLine, tLine)
//    console.log('s2tData=%j', s2tData)
    dataSet.push(s2tData)
  }
  return dataSet
}

LMT.mtTrain = function (text) {
  LMT.words = W.getWordSet(text)
  console.log('LMT.words=%j', LMT.words)
  LMT.w2i = W.words2map(LMT.words)
  console.log('LMT.w2i=%j', LMT.w2i)
  var dataSet = LMT.mtDataSet(text.split(/\r?\n/))
//  console.log('dataSet=%s', JSON.stringify(dataSet))
  LMT.network = LMT.dataTrain(dataSet, LMT.words)
}
