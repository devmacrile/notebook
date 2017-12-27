class Main {

  def pascal(c: Int, r: Int) = {
    def iter(c: Int, r: Int): Int =
      if (c == 0 || c == r) 1
      else iter(c, r - 1) + iter(c - 1, r - 1)
    iter(c, r)
  }


  def balance(chars: List[Char]) = {
    def balance_iter(chars: List[Char], open: Int): Boolean = {
      if (chars.isEmpty && open == 0) true
      else if ((chars.isEmpty && open != 0) || (open >= 0 && chars.head == ')')) false
      else
        chars.head match {
          case '(' => balance_iter(chars.tail, open + 1)
          case ')' => balance_iter(chars.tail, open - 1)
          case _ => balance_iter(chars.tail, open)
        }
    }
    balance_iter(chars, 0)
  }

  def countChange(money: Int, coins: List[Int]): Int = {
    if (money == 0) 0
    else if (money < 0 || coins.isEmpty) 0
    else
      countChange(money - coins.head, coins) + countChange(money, coins.tail)
  }

}
