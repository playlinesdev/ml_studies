require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs-node')
const AI2 = require('./ai2')

const ai2 = new AI2()
ai2.getModel().then(async (model) => {
    const prediction = await model.predict(tf.tensor1d([3, 18]));
    console.log(prediction.print())
}).catch((error) => console.log(error)).finally(() => {
    console.log('FIM!')
})