class Main {

  def pascal(c: Int, r: Int) = {
    def iter(c: Int, r: Int): Int =
      if (c == 0 || c == r) 1
      else iter(c, r - 1) + iter(c - 1, r - 1)
    iter(c, r)
  }

}
