#include <topological_sort.h>
#include <parameter.h>
#include <unordered_set>
#include <iostream>

std::vector<Parameter*> TopologicalSort::run(Parameter* start) {
    std::unordered_set<Parameter*> visited;
    std::vector<Parameter*> order;
    helper(start, visited, order);
    return order;
}

void TopologicalSort::helper(Parameter* start, std::unordered_set<Parameter*>& visited, std::vector<Parameter*>& order) {
    if (!visited.count(start)) {
        visited.insert(start);
        for (int i = 0; i < start->children_.size(); i++) {
            helper(start->children_[i], visited, order);
        }
        order.push_back(start);
    }
}