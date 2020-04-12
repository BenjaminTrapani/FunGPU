#include "Compiler.hpp"
#include "PortableMemPool.hpp"
#include "Types.hpp"

namespace FunGPU {
class BlockPrep {
public:
  BlockPrep(Index_t registersPerBlock, Index_t instructionsPerCycle,
            Index_t cyclesPerBlock);
  Compiler::ASTNodeHandle
  PrepareForBlockGeneration(Compiler::ASTNodeHandle root);

private:
  /*

    Output of this whole process:
    Array of RuntimeBlock instances for each lambda, initial call instruction.

   Wrap the root expression of the program in a lambda taking 0 args. Wrap
   this lambda in a call expression with 0 args. Manually execute this outer
   call instruction in the evaluator.

   For each lambda:
   Generate globally unique integer idx for the lambda (this will be used for
   indirect calls later) Add it to the vector of working programs along with
   the generated ID.

   While working program set > 0:
    For each program:
    Translate program into the following format:
    (lambda (i1 i2...)
     (let ((x1 (prim_op))
       (x2 (prim_op))
        ...)
         (prim_op)))

      prim_op is an identifier, number, lambda or call with all arguments in
      registers already. As a result, callers guarantee that i1, i2 ... are
   prim_ops too.

      Identify number of captured values in lambda. Generate synthetic let
   expressions for the total number of captured values (doesn't matter what is
   bound, since caller will overwrite these). Generate synthetic let bindings
   for each lambda arg too. These will also get overwritten by call
   instruction.

      Then inline all calls where possible. Only expand once.

      Build sequential list of operations per register (each op still writes
   to new register at this point, so each op should just have a write followed
   by one or more reads)

      Walk tree of operations from tail position dependencies backwards.
   Remove unused operations (those writing registers that are not colored in
   this process).

      Start at second to top layer. Move instructions that do not depend on
   first layer up. Repeat recursively. Stop if target layer already has
   m_instructionsPerCycle instructions on it.

      Attempt to perform k-coloring of most recent op tree (registers are
   vertices, dependencies always point up and are dependencies on register
   writes from previous layers). Walk dependencies in reverse. If dependency
   skips a layer, add connections from dependend on register to all registers
   in subsequent layers up to but not including the layer in which the
   dependent register sits. Always add connection from source register to
   dependent. k is m_registersPerBlock.

      If coloring fails or if the number of layers > m_cyclesPerBlock, we've
   inlined too far or the original block is too big. Incrementally remove
   lower layers and retry coloring. Once coloring succeeeds, move overflow
   (lower layers removed) into an indirect call. Arguments are all the
   dependencies of the instructions in the lower layers removed. All remaining
   layers of instructions should be moved into the construction of the new
   synthetic lambda. Synthetic lambda should be created at the top level of the
   current closure. Add lambda to set of pending programs to generate.

      If coloring succeeds, go back to inlining step and repeat. If no
   inlining possible break out of loop. Store coloring, since colors represent
   destination register indices per instruction.

      Target objects are as follows:
      Add:
        Index_t lhs_register_idx;
        Index_t rhs_register_idx;

     Sub:
       Index_t lhs_register_idx;
       Index_t rhs_register_idx;

     Call:
        Index_t arg_registers[m_registersPerBlock];
        Index_t lambda_register_idx;

      Instruction:
        Type type;
        union {
          Add, Sub, ... Call
        } instruction;
        Index_t result_register_idx;

      FunctionVal {
        ArrayHandle<RuntimeValue> closed_values;
        Index_t lambda_block_idx;
      }

      RuntimeValue {
        Type type;
        union {
          float float_value, FunctionVal funv;
        } val;
      }

      RuntimeBlock:
        RuntimeValue registers[m_registersPerBlock];
        Instruction instructions[m_cyclesPerBlock][m_instructionsPerCycle];
        Instruction tail_op;
        Index_t currentCycle = 0;
        const Index_t totalCycles;
        RuntimeValue* destination;
        RuntimeBlock* parent; // to reactivate when indirect call completes.
        Index_t outstandingDependencies = 0;

     One RuntimeBlock created per lambda processed above. Store them in an
   array keyed off of the unique lambda id. In the compilation process, fill
   in the instructions 2D array based on the nested let expressions. Do
   something like this:
   for i in let_blocks:
     for j in let_blocks[i]:
          instructions[i][j] =
   build_instruction_from_primitive_op(let_blocks[i][j]) tail_op =
   build_instruction_from_primitive_op(let_blocks[len(let_blocks) -
   1].bound_expr)

     totalCycles is the total number of cycles required to evaluate each let
   expression and is equal to the total number of nested let blocks.
     destination and parent are initialized at runtime.
     outstandingDependencies = 0 and modified during runtime.

     When lambda is evaluated (lambda instruction encountered), dynamically
   allocate array of runtime values for closed values. Copy the enclosed values
   (specified in lambda instruction) into the array. Attach array handle to
   lambda val in addition to the lambda idx. Store in target register for
   instruction.

     When lambda is called, allocate runtime block for it. Copy closures
   specified in lambda value into the block. Copy call args into register
   range immediately after closures. Set parent to this and destination to
   destination register for call instruction. Add the lambda block as a
   dependency of the current block. It'll get executed in upcoming cycles.
   When a RuntimeBlock is complete (currentCycle == totalCycles), perform tail
   op and write result to destination. Decrement parent
   outstandingDependencies. If decrement causes outstandingDependencies of
   parent to be 0, schedule parent for next pass.

     All other instructions simply deal in local registers and instructions.
   Execute all instructionsPerCycle in parallel, increment currentCycle once
   per iteration. Number of GPU threads per block is equal to
   m_instructionsPerCycle. Load active blocks into shared memory. Then almost
   all ops (non-indirect-calls) will never go to global memory outside of the
   initial load.
  */

  const Index_t m_registersPerBlock;
  const Index_t m_instructionsPerCycle;
  const Index_t m_cyclesPerBlock;
};
} // namespace FunGPU
