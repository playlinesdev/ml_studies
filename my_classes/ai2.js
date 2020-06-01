require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs-node')
module.exports = class AI2 {
    compile() {
        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    units: 1,
                    inputShape: [1]
                }),
                tf.layers.dense({
                    units: 1,
                    inputShape: [1]
                }),
                tf.layers.dense({
                    units: 1,
                    inputShape: [1]
                })
            ]
        })
        model.compile({
            loss: 'meanSquaredLogarithmicError',
            optimizer: 'sgd',
            metrics: ['accuracy']
        })
        return model
    }
    async train(inputArray, outputArray, epochs) {
        const model = this.compile()
        await model.fit(tf.tensor1d(inputArray), tf.tensor1d(outputArray), {
            epochs: epochs,
        }).catch((error) => console.log(error)).then((val) => {
            console.log();
        })
        return model
    }

    async save(model) {
        await model.save('file://./tmp')
    }

    async load() {
        const model = await tf.loadLayersModel('file://./tmp/model.json');
        return model
    }

    predict(model, arrayInput) {
        const prediction = model.predict(tf.tensor1d(arrayInput))
        console.log(prediction.print())
    }

    async getModel() {
        let model = await this.load().catch((error) => { });
        if (!model) {
            model = await this.train([1, 2, 3, 4, 5, 6, 2.5, 15.0, 50, 25], [1, 4, 6, 8, 10, 12, 5, 30.0, 100, 50], 800)
            this.save(model)
        }
        return model
    }
}