// SPDX-FileCopyrightText: 2024 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

//! Module for tree utils like types, traversals.
//!
//! All the traversals implemented in this module should be the same to maintain consistency,
//! which is required for serialisation / deserialisation

/// Intermediary either-like type for implementing [`impl_modify_map_collect`]
#[derive(Clone)]
pub enum ModifyResult<D, N, L> {
    /// Current subtree should be replaced with a node containing the given children,
    /// and an auxiliary data for extra context if needed.
    /// Traversal should continue recursively.
    NodeContinue(D, Vec<N>),
    /// Current subtree is replaced by a leaf containing the given data.
    LeafStop(L),
}

/// Perform generic modify_map_collect
///
/// This is done in 3 steps while traversing the tree in a pre-order DFS traversal:
/// 1. Apply `modify` on current subtree: This operation changes the structure of the current
///    subtree before traversing its children.
/// 2. When encountering leaves, `map` is called to transform a leaf from `A` to `B` type.
///    This is done on children of subtrees which have been traversed after `modify` was called.
/// 3. After modifying & mapping the children of a node, the `collect` method gathers the newly
///    modified & mapped subtrees to create the new subtree.
pub fn impl_modify_map_collect<
    InterimLeafData, // InterimLeafData -> FinalLeafData  when applying `map`
    FinalLeafData,   // [FinalLeafData] -> FinalLeafData when applying `collect`
    AuxTreeData,     // Type of auxiliary data held for a subtree
    InputTree,
    OutputTree: From<FinalLeafData>,
    TreeModifier: FnMut(InputTree) -> ModifyResult<AuxTreeData, InputTree, InterimLeafData>,
    LeafMapper: FnMut(InterimLeafData) -> FinalLeafData,
    Collector: FnMut(AuxTreeData, Vec<OutputTree>) -> OutputTree,
>(
    root: InputTree,
    mut modify: TreeModifier,
    mut map: LeafMapper,
    mut collect: Collector,
) -> OutputTree {
    enum ProcessEvents<ProcessEvent, CollectAuxTreeData> {
        Node(ProcessEvent),
        Collect(CollectAuxTreeData, usize),
    }

    let mut process = vec![ProcessEvents::Node(root)];
    let mut done: Vec<OutputTree> = vec![];

    while let Some(event) = process.pop() {
        match event {
            ProcessEvents::Node(subtree) => match modify(subtree) {
                ModifyResult::LeafStop(data) => {
                    // Instead of pushing a single leaf process on the Process-queue,
                    // map the data and append it directly to the Done-queue
                    done.push(OutputTree::from(map(data)));
                }
                ModifyResult::NodeContinue(node_data, children) => {
                    // the only case where we push further process events in the process queue
                    // We have to first push a collect event to know how many children should be collected when forming back the current subtree
                    process.push(ProcessEvents::Collect(node_data, children.len()));

                    process.extend(
                        children
                            .into_iter()
                            .rev()
                            .map(|child| ProcessEvents::Node(child)),
                    );
                }
            },
            ProcessEvents::Collect(node_data, count) => {
                // We need to reconstruct a subtree which is made of `count` children
                // No panic: We are guaranteed count < done.len() since every Collect(size)
                // corresponds to size nodes pushed to Done-queue
                let children = done.split_off(done.len() - count);
                done.push(collect(node_data, children));
            }
        }
    }

    // No Panic: We only add a single node as root at the beginning of the algorithm
    // which corresponds to this last node in the Done-queue
    done.pop()
        .filter(|_| done.is_empty())
        .expect("Unexpected number of results")
}
