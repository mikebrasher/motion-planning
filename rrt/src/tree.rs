use glam::Vec2;
use std::{error::Error, fmt};

#[derive(Default)]
pub struct Node {
    index: usize,
    position: Vec2,
    parent: Option<usize>,
    children: Vec<usize>,
}

pub struct ArenaTree {
    arena: Vec<Node>,
}

impl ArenaTree {
    pub fn new(position: Vec2) -> Self {
        ArenaTree {
            arena: vec![Node {
                index: 0,
                position,
                ..Default::default()
            }],
        }
    }

    pub fn insert(&mut self, position: Vec2, parent: usize) -> Result<usize, InvalidParentError> {
        let index = self.arena.len();

        // use get to check bounds on parent index
        match self.arena.get_mut(parent) {
            Some(parent_node) => {
                parent_node.children.push(index);

                self.arena.push(Node {
                    index,
                    position,
                    parent: Some(parent),
                    ..Default::default()
                });

                Ok(index)
            }
            None => return Err(InvalidParentError { parent }),
        }
    }

    pub fn closest_node(&self, target: Vec2) -> Option<usize> {
        self.arena
            .iter()
            .map(|node| {
                let dx = target.x - node.position.x;
                let dy = target.y - node.position.y;
                let dist2 = dx * dx + dy * dy;
                (node.index, dist2)
            })
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
    }
}

#[derive(Debug)]
pub struct InvalidParentError {
    parent: usize,
}

impl fmt::Display for InvalidParentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &InvalidParentError { parent } = self;
        write!(f, "invalid parent index for node {parent}")
    }
}

impl Error for InvalidParentError {}
