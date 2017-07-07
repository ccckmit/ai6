var LMT = require('./lstmMt')
var fs = require('fs')

var text = fs.readFileSync(process.argv[2], 'utf8')
LMT.mode = 'word'
// console.log('mt:text=%s', text)
LMT.mtTrain(text)
fs.writeFileSync(process.argv[3], LMT.toJSON())
