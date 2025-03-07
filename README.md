# FunGPU #
This project provides a lightweight compiler and VM that support running a subset of Racket programs on GPUs and other highly parallel compute devices. 

## Supported Targets ##
The VM uses the SYCL API (https://www.khronos.org/sycl/) and any SYCL implementation should work. The current version uses hipSYCL (https://github.com/illuhad/hipSYCL), which uses the clang PTX and HIP backends to generate code for nvidia and  AMD GPUs respectively. The code is known to work for nvidia architecture 61. No additional work should be required to run on AMD GPUs. Intel chips can be targeted using SPIR-V and the ComputeCPP SYCL implementation (https://developer.codeplay.com/products/computecpp/ce/guides/).

## Grammar ##
```
FGPU = Symbol
       Number
       (let ((Symbol <FGPU>)...) <FGPU>) // Bind the value of <FGPU> to "Symbol" for the expression in the body. Evaluates to the body.
       (letrec ((Symbol <FGPU>)...) <FGPU>) // Same as let but bindings can reference themselves.
       (<FGPU> <FGPU>...) // First arg should evaluate to a lambda, remaining expressions are args.
       (if <FGPU> <FGPU> <FGPU>) // If the first expression is non-zero, return the second, else the third.
       (+ <FGPU> <FGPU>) // Arithmetic, all sub-expressions expected to evaluate to numeric values.
       (- <FGPU> <FGPU>)
       (* <FGPU> <FGPU>)
       (/ <FGPU> <FGPU>)
       (= <FGPU> <FGPU>) // comparison, not assignment. 1 if true, 0 if false.
       (eq? <FGPU> <FGPU>) // Functionally identical to "=", kept for compat with racket.
       (> <FGPU> <FGPU>)
       (remainder <FGPU> <FGPU>) // fmodf
       (expt <FGPU> <FGPU>) // pow
       (floor <FGPU>)
       (lambda (Symbol...) <FGPU>)
```
Example programs can be found at https://github.com/BenjaminTrapani/FunGPU/tree/master/TestPrograms

## Architecture ##
The program is generated as follows:
1. Convert input program to S-Expression tree
2. Parse into AST, convert identifiers to De Bruijn indices
3. Rewrite program as sequence of primitive expressions (details below)
4. Convert all lambdas to sequences of side-effecting instructions operating on registers.

In step 1, the program is converted from text to a tree of s-expressions. The s-expression grammer is:
SEXPR = Symbol | Number
        (SEXPR...)

In step 2, all symbol identifiers are replaced by an index representing the distance from the top of the identifier stack to the value they refer to in the current binding scope.

Step 3 converts the program into a format that can be used to generate sequences of side-effecting instructions more easily. The following constraints are enforced:
1. All values on the right side of let bindings are primitive operations and are not bindings (arith ops with identifier args)
2. All numeric literals are bound to identifiers before use.
3. All function invocations have all args bound as identifiers.
4. If expressions only appear in the tail position.
5. All arguments to if expressions are primitive operations (identifiers or calls with all args as identifiers).
6. Entire program is wrapped in no-arg lambda.

In step 3, the sequence of simple instructions is converted into a sequence of side-effection instructions operating on registers. One sequence is created
per lambda. When a lambda appears in the body of another, an instruction to copy all captured values into the heap is performed. The lambda value contains a pointer
to these values. On any invocations of the lambda value, the captured values are copied into the first registers, followed by the function args. Register allocation is performed by identifying the last position at which an identifier is read in the current scope and freeing the mapped register idx after the last read. Identifiers are allocated a register after they are bound.

The example below is for https://github.com/BenjaminTrapani/FunGPU/blob/master/test_programs/SimpleLetRec.fgpu:
```
Parsed program (step 1): 
( letrec 
  (
    ( count-to 
      ( lambda 
        ( cur  target )
        ( if 
          ( =  cur  target ) target 
          ( count-to 
            ( +  1  cur ) target )))))
  ( count-to  0  5 ))

Compiled program (step 2): 
(letrec (
(lambda (argCount: 2)
 (if (= (ident: 1) (ident: 0))
  (ident: 0)
  (call (ident: 2) (+ 1 (ident: 1)) (ident: 0) ))))
(call (ident: 0) 0 5 ))
```
```
Program after prep for block generation (step 3): 
(let (
(lambda (argCount: 0)
 (letrec (
  (lambda (argCount: 2)
   (let (
    (= (ident: 1) (ident: 0)))
    (let (
      (lambda (argCount: 0)
       (let (
        1)
        (let (
          (+ (ident: 0) (ident: 3)))
          (call (ident: 5) (ident: 0) (ident: 3) )))))
      (if (ident: 1)
        (ident: 2)
        (call (ident: 0) ))))))
  (let (
    0
    5)
    (call (ident: 2) (ident: 1) (ident: 0) )))))
(call (ident: 0) ))
```
```
Printed program (step 4): 
Lambda 0: 
0: CreateLambda: reg 0 = 1, capture registers 0, 
1: AssignConstant: reg 1 = 0
2: AssignConstant: reg 2 = 5
3: BlockingCallIndirect: reg 3 = call reg 0, arg registers 1, 2, 

Lambda 1: 
0: Equal: reg 3 = 1 == 2
1: CreateLambda: reg 4 = 2, capture registers 0, 1, 2, 
2: If reg 3 goto 3 else 4
3: Assign: reg 5 = 2
4: BlockingCallIndirect: reg 6 = call reg 4, arg registers 

Lambda 2: 
0: AssignConstant: reg 3 = 1
1: Add: reg 4 = 3 + 1
2: BlockingCallIndirect: reg 5 = call reg 0, arg registers 4, 2, 
```

## VM ##
The VM takes an array of lambdas and produces a numeric result from the program. It assumes the first lambda is a no-op entry point. On each eval cycle, pending call requests to all lambdas are grouped by lambda index into blocks of threads that will execute the same instructions. The mapping of work to GPU is then straightforward: each block is 32 threads that will execute the same instructions (all calls to the same lambda). Each thread gets its own set of registers. The block with the instructions and registers is loaded into local (on-chip) memory. Each thread concurrently evaluates the instructions in the context of its register set, which is pre-populated with the captures and call args in that order. Once each thread is complete (either stalled on a pending indirect call or complete), the eval loop stops. If all threads are done, the results are written out and the block is deallocated. Otherwise it is stored to global memory and its dependencies (additional indirect calls to more lambdas) are evaluated. This loop continues until the value for the initial block is ready. This value is the result of the program.


## Performance ##
Numeric integration (https://github.com/BenjaminTrapani/FunGPU/blob/master/test_programs/NumericIntegrationV2.fgpu) obtains a 3x performance improvement over the Racket implementation. Profiling indicates that the GPU has plenty of available compute power and is limited by the amount of on-chip memory. Using a device with a higher ratio of on-chip memory to compute resources or reducing the shared memory requirement will enable orders of magnitude performance improvements. Infinite tail recursion is supported. The best performance is obtained by authoring programs such that a high branching factor is used to fan out work initially and then tail recursion is used at the leaves of the tree once a high degree of parallelism has been achieved.

## Current state ##
The current implementation has two versions: an initial interpreter (binary fungpu) that evaluates the AST directly. Warp divergence is high and execution efficiency is low, but it contains a garbage collector and is very stable. The more recent and fastest version (./Core/evaluator_v2/FunGPUV2) uses the compilation pipeline above and outperforms the racket implementation on all non-trivial programs (performance gains increase as the number of program subtrees increases, think embarrassingly parallel problems with divide and conquer solutions). The evaluator at ./Core/evaluator_v2/FunGPUV2 does not have a garbage collector yet, which is required to correctly deallocate lambda captures. Lambda captures are leaked in that implementation. The remaining work consists of adding a garbage collector for the V2 evaluator, improving GPU utilization by reducing shared memory requirements and adding optimization steps to the compilation pipeline.

## Building ##
1. Install AdaptiveCpp by following the instructions at https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md
2. mkdir build
3. cd build
3. cmake -DCMAKE_CXX_COMPILER=acpp -DCMAKE_BUILD_TYPE=Release ../
6. make

The built binary can be run as ./Core/evaluator_v2/fungpu_v2 ../test_programs/MergeSort.fgpu

## Build ##
[![Build Status](http://gpuandai.com:8081/job/FunGPU/job/master/badge/icon)](http://gpuandai.com:8081/job/FunGPU/job/master/)
