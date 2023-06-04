#pragma once
#include <vector>
#include <parameter.h>
#include <unordered_set>

class TopologicalSort {
public:    
    static std::vector<Parameter*> run(Parameter* start);
private:
    static void helper(Parameter* start, std::unordered_set<Parameter*>& visited, std::vector<Parameter*>& order);
};
