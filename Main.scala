import java.util.Random

import SingleLayerNeuralNetworks.Perceptrons
import util.GaussianDistribution

import scala.util.control.Breaks

object Main {
  def main(args: Array[String]): Unit = {
    val rng = new Random(1234)

    val train_N = 1000
    val test_N = 200
    val nln = 2

    val train_X = Array.ofDim[Double](train_N, nln)
    val train_T = Array.ofDim[Int](train_N)

    val test_X = Array.ofDim[Double](test_N, nln)
    val test_T = Array.ofDim[Int](test_N)
    val predicted_T = Array.ofDim[Int](test_N)

    val epochs = 2000
    val learningRate = 1.0

    val g1 = new GaussianDistribution(-2.0, 1.0, rng)
    val g2 = new GaussianDistribution(2.0, 1.0, rng)

    for (i <- 0 until train_N/2) {
      train_X(i)(0) = g1.random()
      train_X(i)(1) = g2.random()
      train_T(i) = 1
    }

    for (i <- 0 until test_N/2) {
      test_X(i)(0) = g1.random()
      test_X(i)(1) = g2.random()
      test_T(i) = 1
    }

    for (i <- train_N/2 until train_N) {
      train_X(i)(0) = g2.random()
      train_X(i)(1) = g1.random()
      train_T(i) = -1
    }

    for (i <- test_N/2 until test_N) {
      test_X(i)(0) = g2.random()
      test_X(i)(1) = g1.random()
      test_T(i) = -1
    }

    var epoch = 0

    val classifier = new Perceptrons(nln)

    val b = new Breaks()

    b.breakable {
      while (true) {
        var classified_ = 0

        for (i <- 0 until train_N) {
          classified_ = classified_ + classifier.train(train_X(i), train_T(i), learningRate)
        }

        if (classified_ == train_N) {
          b.break
        }

        epoch = epoch + 1

        if (epoch > epochs) {
          b.break
        }
      }
    }

    for (i <- 0 until test_N) {
      predicted_T(i) = classifier.predict(test_X(i))
    }

    val confusionMatrix = Array.ofDim[Int](2, 2)
    var accuracy = 0.0
    var precision = 0.0
    var recall = 0.0

    for (i <- 0 until test_N) {
      if (predicted_T(i) > 0) {
        if (test_T(i) > 0) {
          accuracy = accuracy + 1
          precision = precision + 1
          recall = recall + 1
          confusionMatrix(0)(0) = confusionMatrix(0)(0) + 1
        }
        else {
          confusionMatrix(1)(0) = confusionMatrix(1)(0) + 1
        }
      }
      else {
        if (test_T(i) > 0) {
          confusionMatrix(0)(1) = confusionMatrix(0)(1) + 1
        }
        else {
          accuracy = accuracy + 1
          confusionMatrix(1)(1) = confusionMatrix(1)(1) + 1
        }
      }
    }

    accuracy = accuracy / test_N
    precision = precision / (confusionMatrix(0)(0) + confusionMatrix(1)(0))
    recall = recall / (confusionMatrix(0)(0) + confusionMatrix(0)(1))

    println("----------------------------")
    println("Perceptrons model evaluation")
    println("----------------------------")
    println("Accuracy:", accuracy * 100)
    println("Precision:", precision * 100)
    println("Recall:", recall * 100)

  }
}
