var LSTM = require('./lstm')
var fs = require('fs')

var argv = process.argv
var maxLen = (argv.length >= 4) ? parseInt(argv[3]) : 10
var json = fs.readFileSync(process.argv[2])
// console.log('json=%j', json)
LSTM.fromJSON(json)
var sLines = ['', '', '', '']
console.log('main:mode=%s maxLen=%d', LSTM.mode, maxLen)
LSTM.genLines(sLines, [], maxLen)
