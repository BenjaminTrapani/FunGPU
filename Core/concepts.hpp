#include <concepts>

namespace FunGPU {
template <typename Type, typename... PossibleTypes>
concept OneOf = (std::same_as<Type, PossibleTypes> || ...);
}
