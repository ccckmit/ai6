/* eslint-disable camelcase */
// view-source:view-source:http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html
var convnetjs = require('convnetjs')
/*
function buildNet () {
  var layer_defs = [];
  layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3});
  layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
  layer_defs.push({type:'pool', sx:2, stride:2});
  layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
  layer_defs.push({type:'pool', sx:2, stride:2});
  layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
  layer_defs.push({type:'pool', sx:2, stride:2});
  layer_defs.push({type:'softmax', num_classes:10});

  var net = new convnetjs.Net();
  net.makeLayers(layer_defs);
}
*/
function loadNet (json) {
  var net = new convnetjs.Net();
  net.fromJSON(json);

  var trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:4, l2_decay:0.0001});
  trainer.learning_rate = 0.0001;
  trainer.momentum = 0.9;
  trainer.batch_size = 2;
  trainer.l2_decay = 0.00001;
  return {net: net, trainer: trainer}
}

function predicts (net, sample) {
  var aavg = new convnetjs.Vol(1, 1, num_classes, 0.0)
  // ensures we always have a list, regardless if above returns single item or list
  var xs = [].concat(sample.x)
  var n = xs.length
  for (var i = 0; i < n; i++) {
    var a = net.forward(xs[i])
    aavg.addFrom(a)
  }
  var preds = []
  for (var k = 0; k < aavg.w.length; k++) {
    preds.push({k: k, p: aavg.w[k]})
  }
  preds.sort(function (a, b) { return a.p < b.p ? 1 : -1 })
}

var sample_test_instance = function(img_data, ) {
  var p = img_data.data;
  var x = new convnetjs.Vol(32,32,3,0.0);
  var W = 32*32;
  var j=0;
  for(var dc=0;dc<3;dc++) {
    var i=0;
    for(var xc=0;xc<32;xc++) {
      for(var yc=0;yc<32;yc++) {
        var ix = i * 4 + dc;
        x.set(yc,xc,dc,p[ix]/255.0-0.5);
        i++;
      }
    }
  }

  // distort position and maybe flip
  var xs = []
  //xs.push(x, 32, 0, 0, false); // push an un-augmented copy
  for (var k = 0; k < 6; k++) {
    var dx = Math.floor(Math.random()*5-2);
    var dy = Math.floor(Math.random()*5-2);
    xs.push(convnetjs.augment(x, 32, dx, dy, k>2));
  }

  // return multiple augmentations, and we will average the network over them
  // to increase performance
  return {x: xs, label: labels[n]}
}

var classes_txt = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
var num_classes = classes_txt.length // net.layers[net.layers.length-1].out_depth;

var pretrainedJson = require('./cifar10_snapshot.json')

var {net, trainer} = loadNet(pretrainedJson)

var sample = {}

predict(sample)

/*
    var sample = sample_test_instance();
    var y = sample.label;  // ground truth label

    // forward prop it through the network
    var aavg = new convnetjs.Vol(1,1,num_classes,0.0);
    // ensures we always have a list, regardless if above returns single item or list
    var xs = [].concat(sample.x);
    var n = xs.length;
    for(var i=0;i<n;i++) {
      var a = net.forward(xs[i]);
      aavg.addFrom(a);
    }
    var preds = [];
    for(var k=0;k<aavg.w.length;k++) { preds.push({k:k,p:aavg.w[k]}); }
    preds.sort(function(a,b){return a.p<b.p ? 1:-1;});
    
    var correct = preds[0].k===y;
    if(correct) num_correct++;
    num_total++;
*/    

// var trainer = new convnetjs.SGDTrainer(net, {method:'adadelta', batch_size:4, l2_decay:0.0001});
// var trainer = new convnetjs.SGDTrainer(net, {learning_rate:trainer.learning_rate, momentum:trainer.momentum, batch_size:trainer.batch_size, l2_decay:trainer.l2_decay});

/*
var num_batches = 51; // 20 training batches, 1 test
var test_batch = 50;
var data_img_elts = new Array(num_batches);
var img_data = new Array(num_batches);
var loaded = new Array(num_batches);
var loaded_train_batches = [];

var use_validation_data = true;

var sample_training_instance = function() {
  // find an unloaded batch
  var bi = Math.floor(Math.random()*loaded_train_batches.length);
  var b = loaded_train_batches[bi];
  var k = Math.floor(Math.random()*1000); // sample within the batch
  var n = b*1000+k;
  // load more batches over time
  if(step_num%2000===0 && step_num>0) {
    for(var i=0;i<num_batches;i++) {
      if(!loaded[i]) {
        // load it
        load_data_batch(i);
        break; // okay for now
      }
    }
  }

  // fetch the appropriate row of the training image and reshape into a Vol
  var p = img_data[b].data;
  var x = new convnetjs.Vol(32,32,3,0.0);
  var W = 32*32;
  var j=0;
  for(var dc=0;dc<3;dc++) {
    var i=0;
    for(var xc=0;xc<32;xc++) {
      for(var yc=0;yc<32;yc++) {
        var ix = ((W * k) + i) * 4 + dc;
        x.set(yc,xc,dc,p[ix]/255.0-0.5);
        i++;
      }
    }
  }
  var dx = Math.floor(Math.random()*5-2);
  var dy = Math.floor(Math.random()*5-2);
  x = convnetjs.augment(x, 32, dx, dy, Math.random()<0.5); //maybe flip horizontally

  var isval = use_validation_data && n%10===0 ? true : false;
  return {x:x, label:labels[n], isval:isval};
}

// 圖片在此： 但看來怪怪的 ....，這看來是作者為了加速，把 1000 張 32*32 的圖片合併成一張載入。
//   http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10/cifar10_batch_50.png
//   http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10/cifar10_batch_0.png
// 轉換公式如下： ((W * k) + i) * 4 + dc;  (W:32*32, k:第k張圖, i:第 i 個點， dc: 第 dc 個像素)
// sample a random testing instance
var sample_test_instance = function() {
  var b = test_batch;
  var k = Math.floor(Math.random()*1000);
  var n = b*1000+k;

  var p = img_data[b].data;
  var x = new convnetjs.Vol(32,32,3,0.0);
  var W = 32*32;
  var j=0;
  for(var dc=0;dc<3;dc++) {
    var i=0;
    for(var xc=0;xc<32;xc++) {
      for(var yc=0;yc<32;yc++) {
        var ix = ((W * k) + i) * 4 + dc;
        x.set(yc,xc,dc,p[ix]/255.0-0.5);
        i++;
      }
    }
  }

  // distort position and maybe flip
  var xs = [];
  //xs.push(x, 32, 0, 0, false); // push an un-augmented copy
  for(var k=0;k<6;k++) {
    var dx = Math.floor(Math.random()*5-2);
    var dy = Math.floor(Math.random()*5-2);
    xs.push(convnetjs.augment(x, 32, dx, dy, k>2));
  }
  
  // return multiple augmentations, and we will average the network over them
  // to increase performance
  return {x:xs, label:labels[n]};
}
*/