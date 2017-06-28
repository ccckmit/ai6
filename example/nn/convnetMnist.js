/* eslint-disable camelcase */
// view-source:http://cs.stanford.edu/people/karpathy/convnetjs/demo/regression.html
var convnetjs = require('convnetjs')

var classes_txt = ['0','1','2','3','4','5','6','7','8','9'];

function buildNet () {
  var dataset_name = "mnist";
  var num_batches = 21; // 20 training batches, 1 test
  var test_batch = 20;
  var num_samples_per_batch = 3000;
  var image_dimension = 28;
  var image_channels = 1;
  var use_validation_data = true;
  var random_flip = false;
  var random_position = false;
  var layer_defs, net, trainer;
  var layer_defs = [];
  layer_defs.push({type:'input', out_sx:24, out_sy:24, out_depth:1});
  layer_defs.push({type:'conv', sx:5, filters:8, stride:1, pad:2, activation:'relu'});
  layer_defs.push({type:'pool', sx:2, stride:2});
  layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
  layer_defs.push({type:'pool', sx:3, stride:3});
  layer_defs.push({type:'softmax', num_classes:10});
  var net = new convnetjs.Net();
  net.makeLayers(layer_defs);
  return net
}

var trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:20, l2_decay:0.001});


