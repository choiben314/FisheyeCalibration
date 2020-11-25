#pragma once

//System Includes
#include <vector>
#include <deque>
#include <list>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>

//Eigen Includes
#include "eigen-3.3.8/Eigen/Core"
#include "eigen-3.3.8/Eigen/StdVector"
#include "eigen-3.3.8/Eigen/Geometry"

namespace std {
	template <typename T> using Evector = vector<T, Eigen::aligned_allocator<T>>;
	template <typename T> using Edeque = deque<T, Eigen::aligned_allocator<T>>;
	template <typename T> using Elist = list<T, Eigen::aligned_allocator<T>>;

	template <typename T, typename C> using Eset = set<T, C, Eigen::aligned_allocator<T>>;
	template <typename T,
		typename H = std::hash<T>,
		typename E = std::equal_to<T>>
		using Eunordered_set = unordered_set<T, H, E, Eigen::aligned_allocator<T>>;

	template <typename TK, typename TV, typename C> using Emap = map<TK, TV, C, Eigen::aligned_allocator<TV>>;

	/*template <typename TK,
			  typename TV,
			  typename H = std::hash<TK>,
			  typename E = std::equal_to<TK>>
	using Eunordered_map = unordered_map<TK, TV, H, E, Eigen::aligned_allocator<TV>>;*/

	template <typename TK,
		typename TV,
		typename H = std::hash<TK>,
		typename E = std::equal_to<TK>>
		using Eunordered_map = unordered_map<TK, TV, H, E, Eigen::aligned_allocator<std::pair<const TK, TV>>>;
}