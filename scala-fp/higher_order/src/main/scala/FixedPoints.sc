import math.abs

object FixedPoints {

  val tolerance = 0.0001
  def isCloseEnough(x: Double, y: Double): Boolean = {
    abs((x - y) / x) / x < tolerance
  }
  def fixedPoint(f: Double => Double)(firstGuess: Double) = {
    def iterate(guess: Double): Double = {
      val next = f(guess)
      if (isCloseEnough(guess, next)) guess
      else iterate(next)
    }
    iterate(firstGuess)
  }

  fixedPoint(x => 1 + x / 2)(1)


  def averageDamp(f: Double => Double)(x: Double) = (x + f(x)) / 2

  def sqrt(x: Double) = {
    fixedPoint(averageDamp(y => x / y))(1.0)
  }

  sqrt(2)
}