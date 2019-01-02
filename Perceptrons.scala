package SingleLayerNeuralNetworks

import util._

class Perceptrons(val nln: Int, val w: Array[Double]) {

  def this(nln: Int) {
    this(nln, new Array[Double](nln))
  }

  def train(x: Array[Double], t: Int, learningRate: Double): Int = {
    var classified = 0
    var c = 0.0

    for (i <- 0 until nln) {
      c = c + w(i) * x(i) * t
    }

    if (c > 0) {
      classified = 1
    }
    else {
      for (i <- 0 until nln) {
        w(i) = w(i) + learningRate * x(i) * t
      }
    }

    classified
  }

  def predict (x: Array[Double]): Int = {
    var preActivation = 0.0

    for (i <- 0 until nln) {
      preActivation = preActivation + w(i) * x(i)
    }

    ActivationFunction.step(preActivation)
  }
}
