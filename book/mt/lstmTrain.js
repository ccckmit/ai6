var LSTM = require('./lstm')
var fs = require('fs')

var text = fs.readFileSync(process.argv[2], 'utf8')
console.log('main:text=%s', text)
LSTM.mode = (process.argv.length > 4) ? process.argv[4] : 'char'
console.log('LSTM.seqTrain=', LSTM.seqTrain)
LSTM.seqTrain(text)
fs.writeFileSync(process.argv[3], LSTM.toJSON())
