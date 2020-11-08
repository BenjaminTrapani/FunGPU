#define BOOST_TEST_MODULE BlockGeneratorTestsModule
#include <boost/test/included/unit_test.hpp>

#include "Core/EvaluatorV2/BlockGenerator.h"
#include "Core/BlockPrep.hpp"
#include "Core/Parser.hpp"
#include "Core/Compiler.hpp"
#include "Core/Parser.hpp"
#include "Core/Visitor.hpp"
#include <stdexcept>

namespace FunGPU::EvaluatorV2 {
  namespace {
    constexpr Index_t REGISTERS_PER_BLOCK = 64;

    struct Fixture {
      void check_program_generates_instructions(const std::string& program_path, const std::vector<std::vector<Instruction>>& instructions) {
        Parser parser(program_path);
        auto parsed_result = parser.ParseProgram();
        std::cout << "Parsed program: " << std::endl;
        parsed_result->DebugPrint(0);
        std::cout << std::endl;

        Compiler compiler(parsed_result, mem_pool_buffer);
        auto compiled_result = compiler.Compile();
        std::cout << "Compiled program: " << std::endl;
        compiler.DebugPrintAST(compiled_result);
        std::cout << std::endl;

        compiled_result = block_prep.PrepareForBlockGeneration(compiled_result);
        std::cout << "Program after prep for block generation: " << std::endl;
        compiler.DebugPrintAST(compiled_result);
        std::cout << std::endl;

        const auto program = block_generator.construct_blocks(compiled_result);
        auto mem_pool_acc = mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
        std::cout << "Printed program: " << std::endl;
        std::cout << print(program, mem_pool_acc);
        
        BOOST_REQUIRE_EQUAL(instructions.size(), program.GetCount());
        for (Index_t lambda_idx = 0; lambda_idx < program.GetCount(); ++lambda_idx) {
          const auto& lambda = mem_pool_acc[0].derefHandle(program)[lambda_idx];
          BOOST_REQUIRE_EQUAL(instructions[lambda_idx].size(), lambda.instructions.GetCount());
          for (Index_t instruction_idx = 0; instruction_idx < lambda.instructions.GetCount(); ++instruction_idx) {
            BOOST_CHECK(instructions[lambda_idx][instruction_idx].equals(mem_pool_acc[0].derefHandle(lambda.instructions)[instruction_idx], mem_pool_acc));
          }
        }
      }

      std::shared_ptr<PortableMemPool> mem_pool_data = std::make_shared<PortableMemPool>();
      cl::sycl::buffer<PortableMemPool> mem_pool_buffer{mem_pool_data, cl::sycl::range<1>(1)};
      BlockPrep block_prep{REGISTERS_PER_BLOCK, 32, 32, mem_pool_buffer};
      BlockGenerator block_generator{mem_pool_buffer, REGISTERS_PER_BLOCK};
    };

    template<typename InstructionType>
    Instruction create_instruction(const InstructionType& value) {
      Instruction result;
      result.type = InstructionType::TYPE;
      visit(result, Visitor {
        [&](InstructionType& to_update) {
          to_update = value;
        },
        [&](auto&) {
          throw std::invalid_argument("Unexpected dispatched instruction type");
        }
      }, [](auto&) {
        throw std::invalid_argument("Unexpected instruction type");
      });
      return result;
    }

    BOOST_FIXTURE_TEST_CASE(NoBindingsBasicTest, Fixture) {
      check_program_generates_instructions("../TestPrograms/NoBindings.fgpu", {{
        create_instruction(AssignConstant{0, 4}),
        create_instruction(AssignConstant{1, 2}),
        create_instruction(AssignConstant{2, 3}),
        create_instruction(AssignConstant{3, 4}),
        create_instruction(Div{4, 0, 1}),
        create_instruction(Mul{5, 2, 3}),
        create_instruction(Add{6, 4, 5})
      }});
    }
  }
}
