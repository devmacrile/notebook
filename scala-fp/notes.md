Programming itself influenced by Von Neumann architecture (mutable variables => memory cells, control structures => jumps, etc.).  

In the end, pure imperative programming is limited by the "Von Neuman" bottleneck: 

	One tends to conceptualize data structures word-by-word.  

Higher levels of abstraction are required (think collections, polynomials, shapes, documents) - and theories of these abstractions.  

Theories without mutation (i.e. mathematical theory has no mutation operators, only means of combinations).  


#### Functional programming  

* without mutable variables, assignments, loops, and other imperative control structures (simplistic view)  
* focusing on the functions: functions are first-class; can be produced, consumed, composed.  
* operators to compose functions  


*Tail recursion*: if a function calls itself as its last action, the function's stack frame can be used.  Tail recursive functions are iterative processes.  Echoes of SICP and the 
distinction between procedure and process.  


#### Currying  

Special syntax for defining functions that return functions in Scala:  

	def sum(f: Int => Int)(a: Int, b: Int): Int = 
	  if (a > b) 0 else f(a) + sum(f)(a + 1, b)  

as opposed to:  

	def sum(f: Int => Int): (Int, Int) => Int = {
	  def sumF(a: Int, b: Int): Int = 
	    if (a > b) 0
	    else f(a) + sumF(a + 1, b)
	  sumF
	}  

Why?  

Allows the application of sum to be more flexible/lazy.  For example, if we have a function `cube` that cubes a number,  
then `sum(cube)` would be a valid expression (whereas a, b would have to be passed at that point).  

**Expansion of multiple parameter lists**  

By repeating the process n times  

	def f(args1)...(argsn-1)(argsn) = E  

is shown to be equivalent to  

	def f = (args1 => (args2 => ...(argsn => E)...))  

This is the result due to Haskell Brooks Curry, hence the *currying*.  


#### Language elements and syntax thus far  

Context free syntax in Extended Backus-Naur form (EBNF) for types, expressions, and definitions where:  

  | denotes an alternative,
  [...] an option (0 or 1),
  {...} a repetition (0 or more)  

**Types**  

Type = SimpleType | FunctionType  
FunctionType = SimpleType '=>' Type
             | '(' [Types] ')' '=>' Type
SimpleType = Ident
Types = Type {',', Type}  

A *type* can be:  

  - A numeric type: Int, Double (and Byte, Short, Char, Long, Float)  
  - The Boolean type with the values *true* and *false*  
  - The String type  
  - A function type, like Int => Int, (Int, Int) => Int  

More forms, but these are the basics.  

An *expression* can be any one of: identifier, literal, function application, operator application, 
selection (i.e. math.abs), conditional expression, block, or anonymous function (i.e. x=>x+1).  

A *definition* can be:  

  - A function definition, like `def square(x: Int) = x * x`  
  - A value definition, like `val y = square(2)`  

A *parameter* can be:  

  - A call-by-value parameter, like (x: Int)  
  - A call-by-name parameter, like (y: => Double)  

 
#### Functions and Data  

Scala keeps the names of types and values in *different namespaces*.

Any method with a parameter can be used like an infix operator!  

i.e. `r.add(s)` becomes `r add s`  

The *precedence* of an operator is determined by its first character (so when overloading, single rule that 
determines precendce of operators).  

For example, parenthize the following expression:  

	a + b ^? c ?^ d less a ==> b | c  

Without changing meaning, would be:  

	((a + b) ^? (c ?^d)) less ((a ==> b) | c)  


