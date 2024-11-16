use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{error::Error, fmt};

#[derive(Clone, Default, Debug)]
pub struct Configuration {
    pub x: f64,
    pub y: f64,
}

impl Configuration {
    pub fn new(x: f64, y: f64) -> Self {
        Configuration { x, y }
    }

    pub fn distance(&self, _other: Configuration) -> f64 {
        0.0
    }
}

#[derive(Debug, Default)]
pub struct Vertex {
    index: usize,
    config: Configuration,
    neighbors: Vec<usize>,
}

#[derive(Debug, Default)]
pub struct Edge {
    pub tail: Configuration,
    pub head: Configuration,
}

#[derive(Debug, Default)]
pub struct Graph {
    arena: Vec<Vertex>,
}

impl Graph {
    pub fn add_vertex(&mut self, config: Configuration) -> usize {
        let index = self.arena.len();

        self.arena.push(Vertex {
            index,
            config,
            ..Default::default()
        });

        index
    }

    pub fn add_edge(&mut self, tail: usize, head: usize) -> Result<Edge, InvalidIndexError> {
        // check bounds before pushing any neighbors
        if tail >= self.arena.len() {
            return Err(InvalidIndexError { index: tail });
        }

        if head >= self.arena.len() {
            return Err(InvalidIndexError { index: head });
        }

        // arena contains both vertices, add bidirectional edge
        self.arena[tail].neighbors.push(head);
        self.arena[head].neighbors.push(tail);

        Ok(Edge {
            tail: self.arena[tail].config.clone(),
            head: self.arena[head].config.clone(),
        })
    }

    pub fn get_vertex(&self, index: usize) -> Result<&Vertex, InvalidIndexError> {
        match self.arena.get(index) {
            Some(vertex) => Ok(vertex),
            None => Err(InvalidIndexError { index }),
        }
    }

    pub fn nearest_vertex(&self, target: Configuration) -> Option<usize> {
        self.arena
            .iter()
            .map(|v| {
                let dx = target.x - v.config.x;
                let dy = target.y - v.config.y;
                let dist2 = dx * dx + dy * dy;
                (v.index, dist2)
            })
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
    }
}

#[derive(Debug)]
pub struct InvalidIndexError {
    index: usize,
}

impl fmt::Display for InvalidIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &InvalidIndexError { index } = self;
        write!(f, "invalid vertex index: {index}")
    }
}

impl Error for InvalidIndexError {}

pub struct RapidlyExploringRandomTree {
    rng: StdRng,
    graph: Graph,
    delta_max: f64,
}

impl RapidlyExploringRandomTree {
    pub fn new(seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);

        let graph = Graph {
            arena: vec![Vertex::default()],
        };

        RapidlyExploringRandomTree {
            rng,
            graph,
            delta_max: 3.0,
        }
    }

    pub fn random_configuration(&mut self) -> Configuration {
        let x = 2.0 * self.rng.gen::<f64>() - 1.0;
        let y = 2.0 * self.rng.gen::<f64>() - 1.0;

        Configuration::new(x, y)
    }

    pub fn new_configuration(
        &self,
        near: usize,
        target: Configuration,
    ) -> Result<Configuration, InvalidIndexError> {
        let near = self.graph.get_vertex(near)?;

        // apply non-holonomic constraints
        let config = if near.config.distance(target.clone()) < self.delta_max {
            target
        } else {
            target
        };

        Ok(config)
    }

    pub fn increment(&mut self) -> Result<Edge, InvalidIndexError> {
        let sample = self.random_configuration();
        let parent = self.graph.nearest_vertex(sample.clone()).unwrap();
        let config = self.new_configuration(parent, sample)?;
        let child = self.graph.add_vertex(config.clone());

        self.graph.add_edge(parent, child)
    }
}
