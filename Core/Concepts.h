#include <concepts>

namespace FunGPU {
template <typename Type, typename... PossibleTypes>
concept OneOf = (std::is_same_v<Type, PossibleTypes> || ...);
}
