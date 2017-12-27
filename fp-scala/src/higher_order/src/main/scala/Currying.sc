object Currying {

  def product(f: Int => Int)(a: Int, b: Int): Int =
    if (a > b) 1
    else f(a) * product(f)(a + 1, b)

  product(x => x * x)(3, 4)


  def factorial(n: Int) = product(x => x)(1, n)

  factorial(6)
  factorial(10)


  def mapReduce(f: Int => Int, combine: (Int, Int) => Int, zero: Int)(a: Int, b: Int): Int =
    if (a > b) zero
    else combine(f(a), mapReduce(f, combine, zero)(a + 1, b))

  // redefine product with our more general mapReduce
  def product_(f: Int => Int)(a: Int, b: Int): Int = mapReduce(x => x, (x, y) => x * y, 1)(a, b)

}