#define BOOST_TEST_MODULE RuntimeBlockTestsModule
#include "Core/EvaluatorV2/RuntimeValue.h"
#include "Core/PortableMemPool.hpp"
#include "sycl/handler.hpp"
#include "Core/EvaluatorV2/IndirectCallHandler.hpp"
#include "Core/EvaluatorV2/RuntimeBlock.hpp"
#include "Core/EvaluatorV2/CompileProgram.hpp"
#include <boost/test/included/unit_test.hpp>

namespace FunGPU::EvaluatorV2 {
namespace {
  struct Fixture {
    using RuntimeBlockType = RuntimeBlock<64, 32>;
    using IndirectCallHandlerType = IndirectCallHandler<RuntimeBlockType, 128, 128>;
    
    Fixture(const std::string& program_path) : program(compile_program(program_path, RuntimeBlockType::NumThreadsPerBlock, RuntimeBlockType::NumThreadsPerBlock, mem_pool_buffer))  {
      std::cout
        << "Running on "
        << work_queue.get_device().get_info<cl::sycl::info::device::name>()
        << ", block size: " << sizeof(RuntimeBlockType) << std::endl;
    }

    std::shared_ptr<PortableMemPool> mem_pool_data =
      std::make_shared<PortableMemPool>();
    cl::sycl::buffer<PortableMemPool> mem_pool_buffer{mem_pool_data,
                                                      cl::sycl::range<1>(1)};
    const Program program;
    std::shared_ptr<IndirectCallHandlerType> indirect_call_handler = std::make_shared<IndirectCallHandlerType>(mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(), 
      program.GetCount());
    cl::sycl::buffer<IndirectCallHandlerType> indirect_call_handler_buff{indirect_call_handler, cl::sycl::range<1>(1)};
    cl::sycl::queue work_queue;
  };

  struct BasicFixture : public Fixture {
    BasicFixture() : Fixture("./TestPrograms/SimpleCall.fgpu") {}
  };

  BOOST_FIXTURE_TEST_CASE(basic, BasicFixture) {
    // simulate indirect call requested by instruction #2.
    cl::sycl::buffer<PortableMemPool::Handle<RuntimeBlockType>> caller_buf(cl::sycl::range<1>(1));
    work_queue.submit([&](cl::sycl::handler& cgh) {
      auto indirect_call_acc = indirect_call_handler_buff.get_access<cl::sycl::access::mode::read_write>(cgh);
      auto mem_pool_acc = mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>(cgh);
      auto caller_acc = caller_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
      const auto tmp_program = program;
      cgh.single_task<class RequestIndirectCall>([indirect_call_acc, mem_pool_acc, tmp_program, caller_acc] {
        const auto lambda_0 = mem_pool_acc[0].derefHandle(tmp_program)[0];
        const auto mock_caller = mem_pool_acc[0].Alloc<RuntimeBlockType>(lambda_0.instructions);
        caller_acc[0] = mock_caller;
        FunctionValue function_value(0, PortableMemPool::ArrayHandle<RuntimeValue>());
        const auto test_args = mem_pool_acc[0].AllocArray<RuntimeValue>(1);
        RuntimeValue& arg_val = mem_pool_acc[0].derefHandle(test_args)[0];
        arg_val.type = RuntimeValue::Type::FLOAT;
        arg_val.data.float_val = 42;
        indirect_call_acc[0].on_indirect_call(mem_pool_acc, mock_caller, function_value, 1, 2, test_args);
      });
    });

    const auto exec_group = IndirectCallHandlerType::create_block_exec_group(work_queue, mem_pool_buffer, 
      indirect_call_handler_buff, program);
    BOOST_CHECK_EQUAL(1, exec_group.block_descs.GetCount());
    BOOST_CHECK_EQUAL(3, exec_group.max_num_instructions);
    auto mem_pool_acc = mem_pool_buffer.get_access<cl::sycl::access::mode::read_write>();
    const auto& block_meta = mem_pool_acc[0].derefHandle(exec_group.block_descs)[0];
    BOOST_REQUIRE(block_meta.instructions != PortableMemPool::ArrayHandle<Instruction>());
    BOOST_REQUIRE(block_meta.block != PortableMemPool::Handle<RuntimeBlockType>());
    const auto& first_block = *mem_pool_acc[0].derefHandle(block_meta.block);
    RuntimeValue expected_value(RuntimeValue(42));
    const auto actual_val = first_block.registers[0][0];
    BOOST_CHECK_EQUAL(static_cast<int>(expected_value.type), static_cast<int>(actual_val.type));
    BOOST_CHECK_EQUAL(expected_value.data.float_val, actual_val.data.float_val);
    BOOST_CHECK(block_meta.instructions == mem_pool_acc[0].derefHandle(program)[0].instructions);
    const auto& target_data = first_block.target_data[0];
    BOOST_CHECK_EQUAL(1, target_data.thread);
    BOOST_CHECK_EQUAL(2, target_data.register_idx);
    BOOST_CHECK(caller_buf.get_access<cl::sycl::access::mode::read>()[0] == target_data.block);
  }
}
}
