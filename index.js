const tf = require('@tensorflow/tfjs-node')

const model = tf.sequential({
    layers: tf.layers.dense({
        units: 1,
        inputShape: [1],
        useBias: true
    })
})