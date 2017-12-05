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


