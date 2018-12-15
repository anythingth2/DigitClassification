const mnist = require('mnist');
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
console.log('start load dataset')
const dataset = mnist.set(10000, 10);
const xTrain = tf.tensor2d(dataset.training.map((v) => v.input));
const yTrain = tf.tensor2d(dataset.training.map((v) => v.output));
console.log('loaded dataset')
const model = tf.sequential();

model.add(tf.layers.reshape({
    inputShape: [28 * 28],
    targetShape: [28, 28, 1]
}));
model.add(tf.layers.conv2d({
    kernelSize: 3,
    strides: 3,
    filters: 8,
    activation: 'relu'
}));
model.add(tf.layers.conv2d({
    kernelSize: 5,
    strides: 3,
    filters: 5,
    activation: 'relu'
}));
model.add(tf.layers.flatten({}))
model.add(tf.layers.dense({
    units: 10,
    activation: 'softmax'
}));


model.compile({
    optimizer: tf.train.sgd(0.1),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
});
console.log('starting fiting')
model.fit(xTrain, yTrain, {
    epochs: 100,
    shuffle: true,
    callbacks: () => {
        console.log('done')
    }
}).then(() => {
    console.log('yay')
});