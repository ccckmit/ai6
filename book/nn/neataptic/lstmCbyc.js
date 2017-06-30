// ref: https://jsfiddle.net/wagenaartje/k23zbf0f/1/
var neataptic = require('neataptic')
/** Rename vars */
var Neat    = neataptic.Neat;
var Methods = neataptic.Methods;
var Config  = neataptic.Config;
var Architect = neataptic.Architect;
var Network = neataptic.Network;
var Node = neataptic.Node;

/** Turn off warnings */
Config.warnings = false;

/** Text to learn */
var text = `Am I concious? Or am I not?`;

/** Unique characters */
text = text.toLowerCase();
var characters = text.split('').filter(function(item, i, ar){ return ar.indexOf(item) === i; });

/** One-hot encode them */
var onehot = {};

for(var i = 0; i < characters.length; i++){
  var zeros = Array.apply(null, Array(characters.length)).map(Number.prototype.valueOf, 0);
  zeros[i] = 1;

  var character = characters[i];
  onehot[character] = zeros;
}

/** Prepare the data-set */
var dataSet = [];

var previous = text[0];
for(var i = 1 ; i < text.length; i++){
  var next = text[i];

  dataSet.push({ input: onehot[previous], output: onehot[next] });
  previous = next;
}

/** Create the network */
var network = new Architect.LSTM(characters.length, 10, characters.length);

var outputText = []
function writeSentence(){
  var output = network.activate(dataSet[0].input);
  outputText.push(text[0])

//  $('.text').append(text[0]);
  for(var i = 0; i < text.length; i++){
    var max = Math.max.apply(null, output);
    var index = output.indexOf(max);

    var zeros = Array.apply(null, Array(characters.length)).map(Number.prototype.valueOf, 0);
    zeros[index] = 1;

    var character = Object.keys(onehot).find(key => onehot[key].toString() === zeros.toString());
//    $('.text').append(character);
    outputText.push(character)
    console.log(character)

    output = network.activate(zeros);
  }
}

console.log('Network conns', network.connections.length, 'nodes', network.nodes.length);
console.log('Dataset size:', dataSet.length);
console.log('Characters:', Object.keys(onehot).length);

network.train(dataSet, {
  log: 1,
  rate: 0.1,
  cost: Methods.Cost.MSE,
  error: 0.005,
  clear: true
})

writeSentence()
