require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs-node')

module.exports = class AI {
    //Compile model
    compile() {
        //Sequential model is a model that outputs all the inputs from one layer to the next 
        //layer sequentially. A simple stack of layers
        const model = tf.sequential();

        //Input layer
        //dense is a type of layer in which every neuron connects to every neurons on the next layer
        model.add(tf.layers.dense({
            units: 3,
            inputShape: [3] //shape of the actual tensor (a tensor with an input of an array)
        }))

        //Output layer
        model.add(tf.layers.dense({
            units: 2 //dimensionality output space
        }))

        //Compiling
        model.compile({
            //risk function which measures the average square difference between estimated values
            //and the actual value
            loss: 'meanSquaredError',
            //sgd = Stochastic Gradient Descent
            optimizer: 'sgd'
        })

        return model;
    }

    //Run model / predict 
    run() {
        var model = this.compile()

        //xs - Input layer
        //three shapes. Each shape contains 3 units
        const xs = tf.tensor2d([
            //after training
            [0.1, 0.2], //this should output [1,0]
            [0.2, 1.0, 0.1], //this should output [0,1]
            [1.0, 1.0, 1.0], //this should output [1,1] 
        ])

        //ys - Output Layer
        const ys = tf.tensor2d([
            [1, 0],
            [0, 1],
            [1, 1],
            [1, 1],
        ])

        //trains the model
        model.fit(xs, ys, {
            epochs: 10000
        }).then(async () => {
            await model.save('file:///./tmp').catch((error) => console.log(error)).then((value) => {
                console.log(value)
            });
            //now, after trained, define data to test it's prediction skills
            const data = tf.tensor2d([
                [1.0, 1.0, 1.0]
            ])

            const prediction = model.predict(data)
            prediction.print()
        })
    }

    async loadAndTest() {
        const model = await tf.loadLayersModel('file:///./tmp/model.json')

        const data = tf.tensor2d([
            [0.1, 0.2, 0.3]
        ])
        const prediction = model.predict(data)
        prediction.print()
    }
}