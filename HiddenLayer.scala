package MultiLayerNeuralNetworks

import java.util.Random
import java.util.function.DoubleFunction
import util._

class HiddenLayer(val nln: Int, val nOut: Int, val W: Array[Double], val b: Array[Double], val rng: Random, val activation: DoubleFunction[Double], val dactivation: DoubleFunction[Double]) {
  def this(nln: Int, nOut: Int, W: Array[Double], b: Array[Double], rng: Random, activation: String) = {
    var rng_ = if (rng == null) {new Random(1234)} else {rng}

    var W_ = if (W == null) {
      var W_tmp = Array.ofDim[Double](nOut, nln)
      var w_ = 1.0 / nln

      for(j <- 0 until nOut) {
        for(i <- 0 until nln) {
          W_tmp(j)(i) = RandomGenerator.uniform(-w_,w_,rng)
        }
      }
      W_tmp
    }
    else {
      W
    }

    var b_ = if (b == null) {new Array[Double](nOut)} else {b}

    var activation_ = if (activation == "sigmoid" || activation == null){
      (x: Double) -> ActivationFunction.sigmoid(x)
    }

    this(nln, nOut, W_, b_, rng_, )


    if (rng == null) {
      rng = new Random(1234)
    }

    if (W == null) {

      var w_ = 1.0 / nln

    }

  }
}
