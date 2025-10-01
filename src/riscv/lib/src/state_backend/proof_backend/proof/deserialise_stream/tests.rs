// SPDX-FileCopyrightText: 2025 TriliTech <contact@trili.tech>
//
// SPDX-License-Identifier: MIT

#[cfg(test)]
mod tag_iter_tests {
    use super::super::TagIter;
    use crate::state_backend::proof_backend::proof::InvalidTagError;
    use crate::state_backend::proof_backend::proof::LeafTag;
    use crate::state_backend::proof_backend::proof::TAG_BLIND;
    use crate::state_backend::proof_backend::proof::TAG_NODE;
    use crate::state_backend::proof_backend::proof::TAG_READ;
    use crate::state_backend::proof_backend::proof::Tag;
    use crate::state_backend::proof_backend::proof::tag_offset;

    #[test]
    fn test_tag_iter_empty() {
        let mut iter = TagIter::new(&[]);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_tag_iter_single_byte_all_valid() {
        // Byte contains: NODE, READ, BLIND, READ (from MSB to LSB)
        let byte = (TAG_NODE << tag_offset(0))
            | (TAG_READ << tag_offset(1))
            | (TAG_BLIND << tag_offset(2))
            | (TAG_READ << tag_offset(3));
        let data = [byte];

        let mut iter = TagIter::new(&data);

        assert_eq!(iter.next(), Some(Ok(Tag::Node)));
        assert_eq!(iter.next(), Some(Ok(Tag::Leaf(LeafTag::Read))));
        assert_eq!(iter.next(), Some(Ok(Tag::Leaf(LeafTag::Blind))));
        assert_eq!(iter.next(), Some(Ok(Tag::Leaf(LeafTag::Read))));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_tag_iter_multiple_bytes() {
        // First byte has NODE, READ, and 2 more tags (filled with 0)
        // The TAG_NODE is 0b00, TAG_READ is 0b11
        // A byte with only 2 filled tags should have the remaining filled with TAG_NODE (0)
        let byte1 = (TAG_NODE << tag_offset(0))
            | (TAG_READ << tag_offset(1))
            | (TAG_NODE << tag_offset(2))
            | (TAG_NODE << tag_offset(3));
        let byte2 = (TAG_BLIND << tag_offset(0))
            | (TAG_NODE << tag_offset(1))
            | (TAG_NODE << tag_offset(2))
            | (TAG_NODE << tag_offset(3));
        let data = [byte1, byte2];

        let mut iter = TagIter::new(&data);

        assert_eq!(iter.next(), Some(Ok(Tag::Node)));
        assert_eq!(iter.next(), Some(Ok(Tag::Leaf(LeafTag::Read))));
        assert_eq!(iter.next(), Some(Ok(Tag::Node)));
        assert_eq!(iter.next(), Some(Ok(Tag::Node)));
        assert_eq!(iter.next(), Some(Ok(Tag::Leaf(LeafTag::Blind))));
        assert_eq!(iter.next(), Some(Ok(Tag::Node)));
        assert_eq!(iter.next(), Some(Ok(Tag::Node)));
        assert_eq!(iter.next(), Some(Ok(Tag::Node)));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_tag_iter_invalid_tag() {
        // Invalid tag value 0b01 at position 1
        let byte = (TAG_NODE << tag_offset(0)) | (0b01 << tag_offset(1));
        let data = [byte];

        let mut iter = TagIter::new(&data);

        assert_eq!(iter.next(), Some(Ok(Tag::Node)));
        assert!(matches!(iter.next(), Some(Err(InvalidTagError))));
    }

    #[test]
    fn test_tag_iter_remaining_to_stream_input() {
        let byte1 = TAG_NODE << tag_offset(0);
        let remaining_data = [1u8, 2, 3, 4];
        let all_data = [&[byte1], remaining_data.as_ref()].concat();

        let mut iter = TagIter::new(&all_data);

        // Consume one tag
        let _ = iter.next();

        // Get remaining stream input
        let mut stream_input = iter.remaining_to_stream_input();

        // The remaining data should start after the tag byte
        let mut buffer = vec![0u8; 4];
        let result = std::io::Read::read(&mut stream_input.cursor, &mut buffer);

        // Should read all 4 remaining data bytes
        assert!(result.is_ok());
        assert_eq!(&buffer[..], &remaining_data);
    }

    #[test]
    fn test_tag_iter_all_same_tag() {
        let byte = (TAG_BLIND << tag_offset(0))
            | (TAG_BLIND << tag_offset(1))
            | (TAG_BLIND << tag_offset(2))
            | (TAG_BLIND << tag_offset(3));
        let data = [byte];

        let mut iter = TagIter::new(&data);

        for _ in 0..4 {
            assert_eq!(iter.next(), Some(Ok(Tag::Leaf(LeafTag::Blind))));
        }
        assert!(iter.next().is_none());
    }
}

#[cfg(test)]
mod stream_input_tests {
    use std::io::Cursor;

    use super::super::StreamInput;
    use crate::storage::DIGEST_SIZE;
    use crate::storage::Hash;

    #[test]
    fn test_deserialise_primitive_types() {
        let data = 42i32.to_le_bytes();
        let mut input = StreamInput {
            cursor: Cursor::new(data.as_ref()),
        };

        let result: i32 = input.deserialise().unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_deserialise_bool() {
        let data = [1u8];
        let mut input = StreamInput {
            cursor: Cursor::new(data.as_ref()),
        };

        let result: bool = input.deserialise().unwrap();
        assert_eq!(result, true);

        let data = [0u8];
        let mut input = StreamInput {
            cursor: Cursor::new(data.as_ref()),
        };

        let result: bool = input.deserialise().unwrap();
        assert_eq!(result, false);
    }

    #[test]
    fn test_deserialise_hash() {
        let hash_bytes: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[1, 2, 3]).into();
        let mut input = StreamInput {
            cursor: Cursor::new(hash_bytes.as_ref()),
        };

        let result: Hash = input.deserialise().unwrap();
        assert_eq!(result, Hash::blake3_hash_bytes(&[1, 2, 3]));
    }

    #[test]
    fn test_deserialise_insufficient_data() {
        // Try to deserialize i32 from only 2 bytes
        let data = [1u8, 2];
        let mut input = StreamInput {
            cursor: Cursor::new(data.as_ref()),
        };

        let result: Result<i32, _> = input.deserialise();
        assert!(result.is_err());
    }

    #[test]
    fn test_deserialise_multiple_values() {
        let data = [42i32.to_le_bytes().as_ref(), 100i32.to_le_bytes().as_ref()].concat();

        let mut input = StreamInput {
            cursor: Cursor::new(data.as_ref()),
        };

        let first: i32 = input.deserialise().unwrap();
        let second: i32 = input.deserialise().unwrap();

        assert_eq!(first, 42);
        assert_eq!(second, 100);
    }

    #[test]
    fn test_cursor_position_tracking() {
        let data = [1u8, 2, 3, 4, 5];
        let mut input = StreamInput {
            cursor: Cursor::new(data.as_ref()),
        };

        assert_eq!(input.cursor.position(), 0);

        let _: u8 = input.deserialise().unwrap();
        assert_eq!(input.cursor.position(), 1);

        let _: u8 = input.deserialise().unwrap();
        assert_eq!(input.cursor.position(), 2);
    }
}

#[cfg(test)]
mod stream_deserialiser_tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::super::StreamDeserialiser;
    use super::super::TagIter;
    use super::super::deserialiser::Deserialiser;
    use super::super::deserialiser::DeserialiserNode;
    use super::super::deserialiser::Partial;
    use crate::state_backend::ProofError;
    use crate::state_backend::proof_backend::proof::TAG_BLIND;
    use crate::state_backend::proof_backend::proof::TAG_NODE;
    use crate::state_backend::proof_backend::proof::TAG_READ;
    use crate::state_backend::proof_backend::proof::tag_offset;
    use crate::storage::DIGEST_SIZE;
    use crate::storage::Hash;

    #[test]
    fn test_absent_deserialiser_into_leaf() {
        let deser = StreamDeserialiser::Absent;

        let suspended = deser.into_leaf::<i32>().unwrap();

        let empty_data = [];
        let tag_iter = TagIter::new(&empty_data);
        let mut empty_input = tag_iter.remaining_to_stream_input();
        let result = suspended.into_result(&mut empty_input).unwrap();

        assert!(matches!(result, Partial::Absent));
    }

    #[test]
    fn test_absent_deserialiser_into_leaf_raw() {
        let deser = StreamDeserialiser::Absent;

        let suspended = deser.into_leaf_raw::<32>().unwrap();

        let empty_data = [];
        let tag_iter = TagIter::new(&empty_data);
        let mut empty_input = tag_iter.remaining_to_stream_input();
        let result = suspended.into_result(&mut empty_input).unwrap();

        assert!(matches!(result, Partial::Absent));
    }

    #[test]
    fn test_absent_deserialiser_into_node() {
        let deser = StreamDeserialiser::Absent;

        let node = deser.into_node().unwrap().done().unwrap();

        let empty_data = [];
        let tag_iter = TagIter::new(&empty_data);
        let mut empty_input = tag_iter.remaining_to_stream_input();
        let result = node.into_result(&mut empty_input).unwrap();

        assert!(matches!(result, Partial::Absent));
    }

    #[test]
    fn test_present_deserialiser_leaf_read() {
        let tag_byte = TAG_READ << tag_offset(0);
        let data = 42i32.to_le_bytes();
        let all_data = [&[tag_byte], data.as_ref()].concat();

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let deser = StreamDeserialiser::new_present(tags.clone());

        let suspended = deser.into_leaf::<i32>().unwrap();

        let binding = tags.borrow_mut();
        let mut stream_input = binding.remaining_to_stream_input();
        let result = suspended.into_result(&mut stream_input).unwrap();

        match result {
            Partial::Present((value, raw_bytes)) => {
                assert_eq!(value, 42);
                assert_eq!(raw_bytes, data.to_vec());
            }
            _ => panic!("Expected Present variant"),
        }
    }

    #[test]
    fn test_present_deserialiser_leaf_blind() {
        let tag_byte = TAG_BLIND << tag_offset(0);
        let hash_bytes: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[1, 2, 3]).into();
        let all_data = [&[tag_byte], hash_bytes.as_ref()].concat();

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let deser = StreamDeserialiser::new_present(tags.clone());

        let suspended = deser.into_leaf::<i32>().unwrap();

        let binding = tags.borrow_mut();
        let mut stream_input = binding.remaining_to_stream_input();
        let result = suspended.into_result(&mut stream_input).unwrap();

        match result {
            Partial::Blinded(hash) => {
                assert_eq!(hash, Hash::blake3_hash_bytes(&[1, 2, 3]));
            }
            _ => panic!("Expected Blinded variant"),
        }
    }

    #[test]
    fn test_present_deserialiser_leaf_raw_read() {
        let tag_byte = TAG_READ << tag_offset(0);
        let data = [42u8; 32];
        let all_data = [&[tag_byte], data.as_ref()].concat();

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let deser = StreamDeserialiser::new_present(tags.clone());

        let suspended = deser.into_leaf_raw::<32>().unwrap();

        let binding = tags.borrow_mut();
        let mut stream_input = binding.remaining_to_stream_input();
        let result = suspended.into_result(&mut stream_input).unwrap();

        match result {
            Partial::Present(boxed_data) => {
                assert_eq!(*boxed_data, data);
            }
            _ => panic!("Expected Present variant"),
        }
    }

    #[test]
    fn test_present_deserialiser_leaf_raw_insufficient_bytes() {
        let tag_byte = TAG_READ << tag_offset(0);
        let data = [42u8; 16]; // Only 16 bytes, but we expect 32
        let all_data = [&[tag_byte], data.as_ref()].concat();

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let deser = StreamDeserialiser::new_present(tags.clone());

        let suspended = deser.into_leaf_raw::<32>().unwrap();

        let binding = tags.borrow_mut();
        let mut stream_input = binding.remaining_to_stream_input();
        let result = suspended.into_result(&mut stream_input);

        assert!(result.is_err());
    }

    #[test]
    fn test_present_deserialiser_node_tag() {
        let tag_byte = TAG_NODE << tag_offset(0);
        let all_data = [tag_byte];

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let deser = StreamDeserialiser::new_present(tags.clone());

        let node = deser.into_node().unwrap().done().unwrap();

        let binding = tags.borrow_mut();
        let mut stream_input = binding.remaining_to_stream_input();
        let result = node.into_result(&mut stream_input).unwrap();

        assert!(matches!(result, Partial::Present(())));
    }

    #[test]
    fn test_present_deserialiser_node_blind() {
        let tag_byte = TAG_BLIND << tag_offset(0);
        let hash_bytes: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[5, 6, 7]).into();
        let all_data = [&[tag_byte], hash_bytes.as_ref()].concat();

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let deser = StreamDeserialiser::new_present(tags.clone());

        let node = deser.into_node().unwrap().done().unwrap();

        let binding = tags.borrow_mut();
        let mut stream_input = binding.remaining_to_stream_input();
        let result = node.into_result(&mut stream_input).unwrap();

        match result {
            Partial::Blinded(hash) => {
                assert_eq!(hash, Hash::blake3_hash_bytes(&[5, 6, 7]));
            }
            _ => panic!("Expected Blinded variant"),
        }
    }

    #[test]
    fn test_present_deserialiser_unexpected_node() {
        let tag_byte = TAG_NODE << tag_offset(0);
        let all_data = [tag_byte];

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let deser = StreamDeserialiser::new_present(tags);

        let result = deser.into_leaf::<i32>();

        assert!(matches!(result, Err(ProofError::UnexpectedNode)));
    }

    #[test]
    fn test_present_deserialiser_unexpected_leaf() {
        let tag_byte = TAG_READ << tag_offset(0);
        let all_data = [tag_byte];

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let deser = StreamDeserialiser::new_present(tags);

        let result = deser.into_node();

        assert!(matches!(result, Err(ProofError::UnexpectedLeaf)));
    }
}

#[cfg(test)]
mod stream_parser_comb_tests {
    use super::super::StreamInput;
    use super::super::StreamParserComb;
    use super::super::TagIter;
    use super::super::deserialiser::Partial;
    use super::super::deserialiser::Suspended;
    use crate::state_backend::ProofError;

    #[test]
    fn test_parser_comb_simple_execution() {
        let parser = StreamParserComb::new(|input: &mut StreamInput| {
            let value: i32 = input.deserialise()?;
            Ok(Partial::Present(value))
        });

        let data = 123i32.to_le_bytes();
        let tag_iter = TagIter::new(&data);
        let mut stream_input = tag_iter.remaining_to_stream_input();

        let result = parser.into_result(&mut stream_input).unwrap();

        match result {
            Partial::Present(value) => assert_eq!(value, 123),
            _ => panic!("Expected Present variant"),
        }
    }

    #[test]
    fn test_parser_comb_map() {
        let parser = StreamParserComb::new(|input: &mut StreamInput| {
            let value: i32 = input.deserialise()?;
            Ok(value)
        });

        let mapped = parser.map(|x| x * 2);

        let data = 21i32.to_le_bytes();
        let tag_iter = TagIter::new(&data);
        let mut stream_input = tag_iter.remaining_to_stream_input();

        let result = mapped.into_result(&mut stream_input).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_parser_comb_chain_maps() {
        let parser = StreamParserComb::new(|input: &mut StreamInput| {
            let value: i32 = input.deserialise()?;
            Ok(value)
        });

        let mapped = parser.map(|x| x + 10).map(|x| x * 2).map(|x| x - 4);

        let data = 5i32.to_le_bytes();
        let tag_iter = TagIter::new(&data);
        let mut stream_input = tag_iter.remaining_to_stream_input();

        let result = mapped.into_result(&mut stream_input).unwrap();
        assert_eq!(result, (5 + 10) * 2 - 4);
    }

    #[test]
    fn test_into_result_success_no_remaining_bytes() {
        let parser = StreamParserComb::new(|input: &mut StreamInput| {
            let value: i32 = input.deserialise()?;
            Ok(value)
        });

        let data = 42i32.to_le_bytes();
        let tag_iter = TagIter::new(&data);
        let mut stream_input = tag_iter.remaining_to_stream_input();

        let result = parser.into_result(&mut stream_input);

        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_into_result_fails_with_remaining_bytes() {
        let parser = StreamParserComb::new(|input: &mut StreamInput| {
            let value: i32 = input.deserialise()?;
            Ok(value)
        });

        let data = [42i32.to_le_bytes().as_ref(), &[99u8]].concat();
        let tag_iter = TagIter::new(&data);
        let mut stream_input = tag_iter.remaining_to_stream_input();

        let result = parser.into_result(&mut stream_input);

        assert!(matches!(result, Err(ProofError::RemainingBytes)));
    }

    #[test]
    fn test_into_result_propagates_error() {
        let parser: StreamParserComb<i32> =
            StreamParserComb::new(|_input: &mut StreamInput| Err(ProofError::UnexpectedNode));

        let data = [];
        let tag_iter = TagIter::new(&data);
        let mut stream_input = tag_iter.remaining_to_stream_input();

        let result = parser.into_result(&mut stream_input);

        assert!(matches!(result, Err(ProofError::UnexpectedNode)));
    }
}

#[cfg(test)]
mod stream_branch_comb_tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::super::StreamDeserialiser;
    use super::super::TagIter;
    use super::super::deserialiser::Deserialiser;
    use super::super::deserialiser::DeserialiserNode;
    use super::super::deserialiser::Partial;
    use crate::state_backend::proof_backend::proof::TAG_NODE;
    use crate::state_backend::proof_backend::proof::TAG_READ;
    use crate::state_backend::proof_backend::proof::tag_offset;

    #[test]
    fn test_branch_comb_next_branch() {
        let tag_bytes = (TAG_NODE << tag_offset(0)) | (TAG_READ << tag_offset(1));
        let leaf_data = 100i32.to_le_bytes();
        let all_data = [&[tag_bytes], leaf_data.as_ref()].concat();

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let deser = StreamDeserialiser::new_present(tags.clone());

        let with_child = deser
            .into_node()
            .unwrap()
            .next_branch(|child_deser| child_deser.into_leaf::<i32>())
            .unwrap()
            .done()
            .unwrap();

        let binding = tags.borrow_mut();
        let mut stream_input = binding.remaining_to_stream_input();
        let result = with_child.into_result(&mut stream_input).unwrap();

        match result {
            (Partial::Present(()), Partial::Present((value, _))) => assert_eq!(value, 100),
            _ => panic!("Expected Present variant with value 100"),
        }
    }

    #[test]
    fn test_branch_comb_map() {
        let tag_bytes = TAG_NODE << tag_offset(0);
        let all_data = [tag_bytes];

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let deser = StreamDeserialiser::new_present(tags.clone());

        let mapped = deser.into_node().unwrap().map(|_| 42).done().unwrap();

        let binding = tags.borrow_mut();
        let mut stream_input = binding.remaining_to_stream_input();
        let result = mapped.into_result(&mut stream_input).unwrap();

        assert_eq!(result, 42);
    }

    #[test]
    fn test_branch_comb_multiple_branches() {
        let tag_bytes =
            (TAG_NODE << tag_offset(0)) | (TAG_READ << tag_offset(1)) | (TAG_READ << tag_offset(2));
        let data1 = 10i32.to_le_bytes();
        let data2 = 20i32.to_le_bytes();
        let all_data = [&[tag_bytes], data1.as_ref(), data2.as_ref()].concat();

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let deser = StreamDeserialiser::new_present(tags.clone());

        let with_branches = deser
            .into_node()
            .unwrap()
            .next_branch(|d| d.into_leaf::<i32>())
            .unwrap()
            .next_branch(|d| d.into_leaf::<i32>())
            .unwrap()
            .done()
            .unwrap();

        let binding = tags.borrow_mut();
        let mut stream_input = binding.remaining_to_stream_input();
        let result = with_branches.into_result(&mut stream_input).unwrap();

        match result {
            ((Partial::Present(()), Partial::Present((v1, _))), Partial::Present((v2, _))) => {
                assert_eq!(v1, 10);
                assert_eq!(v2, 20);
            }
            _ => panic!("Expected both branches to be Present"),
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::super::StreamDeserialiser;
    use super::super::TagIter;
    use super::super::deserialiser::Deserialiser;
    use super::super::deserialiser::DeserialiserNode;
    use super::super::deserialiser::Partial;
    use super::super::deserialiser::Suspended;
    use crate::state_backend::proof_backend::proof::TAG_BLIND;
    use crate::state_backend::proof_backend::proof::TAG_NODE;
    use crate::state_backend::proof_backend::proof::TAG_READ;
    use crate::state_backend::proof_backend::proof::tag_offset;
    use crate::storage::DIGEST_SIZE;
    use crate::storage::Hash;

    fn simple_tree_deserialiser<D: Deserialiser>(
        deser: D,
    ) -> crate::state_backend::proof_backend::proof::deserialiser::Result<
        <D as Deserialiser>::Suspended<(i32, i32)>,
    > {
        // Tree structure: Node with two i32 leaves
        let node = deser.into_node()?;

        let with_children = node
            .next_branch(|d| d.into_leaf::<i32>())
            .unwrap()
            .next_branch(|d| d.into_leaf::<i32>())
            .unwrap()
            .done()?;

        Ok(with_children.map(|((_, left), right)| {
            let l = match left {
                Partial::Present((v, _)) => v,
                Partial::Absent => 0,
                Partial::Blinded(_) => -1,
            };
            let r = match right {
                Partial::Present((v, _)) => v,
                Partial::Absent => 0,
                Partial::Blinded(_) => -1,
            };
            (l, r)
        }))
    }

    #[test]
    fn test_full_deserialization_simple_tree() {
        let tag_bytes =
            (TAG_NODE << tag_offset(0)) | (TAG_READ << tag_offset(1)) | (TAG_READ << tag_offset(2));
        let data1 = 42i32.to_le_bytes();
        let data2 = 100i32.to_le_bytes();
        let all_data = [&[tag_bytes], data1.as_ref(), data2.as_ref()].concat();

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let comp_fn =
            simple_tree_deserialiser(StreamDeserialiser::new_present(tags.clone())).unwrap();

        let binding = tags.borrow_mut();
        let result = comp_fn
            .into_result(&mut binding.remaining_to_stream_input())
            .unwrap();

        assert_eq!(result, (42, 100));
    }

    #[test]
    fn test_full_deserialization_with_blind_leaf() {
        let tag_bytes = (TAG_NODE << tag_offset(0))
            | (TAG_READ << tag_offset(1))
            | (TAG_BLIND << tag_offset(2));
        let data1 = 42i32.to_le_bytes();
        let hash_bytes: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[1, 2, 3]).into();
        let all_data = [&[tag_bytes], data1.as_ref(), hash_bytes.as_ref()].concat();

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let comp_fn =
            simple_tree_deserialiser(StreamDeserialiser::new_present(tags.clone())).unwrap();

        let binding = tags.borrow_mut();
        let result = comp_fn
            .into_result(&mut binding.remaining_to_stream_input())
            .unwrap();

        assert_eq!(result, (42, -1));
    }

    #[test]
    fn test_nested_tree_structure() {
        // Tree: Node -> [Leaf(i32), Node -> [Leaf(i32), Leaf(i32)]]
        // Note: All unused tag positions will be 0 (TAG_NODE) but won't be parsed
        let tag_bytes1 = (TAG_NODE << tag_offset(0))
            | (TAG_READ << tag_offset(1))
            | (TAG_NODE << tag_offset(2))
            | (TAG_READ << tag_offset(3));
        let tag_bytes2 = (TAG_READ << tag_offset(0))
            | (TAG_NODE << tag_offset(1))
            | (TAG_NODE << tag_offset(2))
            | (TAG_NODE << tag_offset(3));

        let data1 = 10i32.to_le_bytes();
        let data2 = 20i32.to_le_bytes();
        let data3 = 30i32.to_le_bytes();

        let all_data = [
            &[tag_bytes1, tag_bytes2],
            data1.as_ref(),
            data2.as_ref(),
            data3.as_ref(),
        ]
        .concat();

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let deser = StreamDeserialiser::new_present(tags.clone());

        let comp_fn = deser
            .into_node()
            .unwrap()
            .next_branch(|d| d.into_leaf::<i32>())
            .unwrap()
            .next_branch(|d| {
                let node = d.into_node()?;
                node.next_branch(|d| d.into_leaf::<i32>())
                    .unwrap()
                    .next_branch(|d| d.into_leaf::<i32>())
                    .unwrap()
                    .done()
            })
            .unwrap()
            .done()
            .unwrap();

        let binding = tags.borrow_mut();
        let result = comp_fn
            .into_result(&mut binding.remaining_to_stream_input())
            .unwrap();

        match result {
            (
                (_, Partial::Present((v1, _))),
                ((_, Partial::Present((v2, _))), Partial::Present((v3, _))),
            ) => {
                assert_eq!(v1, 10);
                assert_eq!(v2, 20);
                assert_eq!(v3, 30);
            }
            _ => panic!("Unexpected result structure"),
        }
    }

    #[test]
    fn test_absent_root() {
        let deser = StreamDeserialiser::Absent;
        let comp_fn = simple_tree_deserialiser(deser).unwrap();

        let empty_data = [];
        let tag_iter = TagIter::new(&empty_data);
        let result = comp_fn
            .into_result(&mut tag_iter.remaining_to_stream_input())
            .unwrap();

        assert_eq!(result, (0, 0));
    }

    #[test]
    fn test_complex_computation_with_mapping() {
        // Sum all present i32 leaves in a flat tree
        let tag_bytes = (TAG_NODE << tag_offset(0))
            | (TAG_READ << tag_offset(1))
            | (TAG_BLIND << tag_offset(2))
            | (TAG_READ << tag_offset(3));

        let data1 = 10i32.to_le_bytes();
        let hash_bytes: [u8; DIGEST_SIZE] = Hash::blake3_hash_bytes(&[9, 9, 9]).into();
        let data2 = 20i32.to_le_bytes();

        let all_data = [
            &[tag_bytes],
            data1.as_ref(),
            hash_bytes.as_ref(),
            data2.as_ref(),
        ]
        .concat();

        let tags = Rc::new(RefCell::new(TagIter::new(&all_data)));
        let deser = StreamDeserialiser::new_present(tags.clone());

        let comp_fn = deser
            .into_node()
            .unwrap()
            .map(|_| Vec::<i32>::new())
            .next_branch(|d| d.into_leaf::<i32>())
            .unwrap()
            .map(|(mut acc, val)| {
                if let Partial::Present((v, _)) = val {
                    acc.push(v);
                }
                acc
            })
            .next_branch(|d| d.into_leaf::<i32>())
            .unwrap()
            .map(|(mut acc, val)| {
                if let Partial::Present((v, _)) = val {
                    acc.push(v);
                }
                acc
            })
            .next_branch(|d| d.into_leaf::<i32>())
            .unwrap()
            .map(|(mut acc, val)| {
                if let Partial::Present((v, _)) = val {
                    acc.push(v);
                }
                acc
            })
            .done()
            .unwrap()
            .map(|vals| vals.into_iter().sum::<i32>());

        let result = comp_fn
            .into_result(&mut tags.borrow_mut().remaining_to_stream_input())
            .unwrap();

        assert_eq!(result, 30); // 10 + 20, blind leaf ignored
    }
}
