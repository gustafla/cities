use ansi_term::{ANSIString, Color};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    Nowhere = 0b0000,
    North = 0b0001,
    East = 0b0010,
    South = 0b0100,
    West = 0b1000,
}

impl Direction {
    pub fn inverse(&self) -> Self {
        match self {
            Self::Nowhere => Self::Nowhere,
            Self::North => Self::South,
            Self::South => Self::North,
            Self::East => Self::West,
            Self::West => Self::East,
        }
    }
}

pub type Node = (usize, usize);
pub type Edge = (Node, Direction);

#[derive(Clone, Copy, Default)]
pub struct EdgeSet(u8);

impl EdgeSet {
    pub fn all() -> Self {
        Self(0b1111)
    }

    pub fn fill(&mut self) {
        self.0 = Self::all().0;
    }

    pub fn add(&mut self, dir: Direction) {
        self.0 |= dir as u8;
    }

    pub fn remove(&mut self, dir: Direction) {
        self.0 &= (!(dir as u8)) & Self::all().0;
    }

    pub fn contains(&self, dir: Direction) -> bool {
        self.0 & dir as u8 != 0
    }
}

impl std::fmt::Display for EdgeSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        let charset = [
            /* 0b0000 */ '·', /* 0b0001 */ '╵', /* 0b0010 */ '╶',
            /* 0b0011 */ '└', /* 0b0100 */ '╷', /* 0b0101 */ '│',
            /* 0b0110 */ '┌', /* 0b0111 */ '├', /* 0b1000 */ '╴',
            /* 0b1001 */ '┘', /* 0b1010 */ '─', /* 0b1011 */ '┴',
            /* 0b1100 */ '┐', /* 0b1101 */ '┤', /* 0b1110 */ '┬',
            /* 0b1111 */ '┼',
        ];
        f.write_char(charset[usize::from(self.0)])
    }
}

#[derive(Debug)]
pub enum GraphOperation {
    InterconnectAll,
    AddBidirectional(Edge),
    RemoveBidirectional(Edge),
}

#[derive(Debug)]
pub struct IndexOutOfBoundsError;

#[derive(Clone)]
pub struct SquareGraph {
    dimension: usize,
    nodes: Vec<EdgeSet>,
    visited: Vec<bool>,
}

impl SquareGraph {
    pub fn index(&self, node: Node) -> Option<usize> {
        let idx = node.0 * self.dimension + node.1;
        if idx >= self.nodes.len() {
            None
        } else {
            Some(idx)
        }
    }

    pub fn node(&self, node: Node) -> Option<&EdgeSet> {
        self.index(node).and_then(|idx| self.nodes.get(idx))
    }

    fn node_mut(&mut self, node: Node) -> Option<&mut EdgeSet> {
        self.index(node).and_then(|idx| self.nodes.get_mut(idx))
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn neighbor(&self, ((y, x), dir): Edge) -> Option<Node> {
        match dir {
            Direction::Nowhere => Some((y, x)),
            Direction::North => {
                if y > 0 {
                    Some((y - 1, x))
                } else {
                    None
                }
            }
            Direction::East => {
                if x < self.dimension - 1 {
                    Some((y, x + 1))
                } else {
                    None
                }
            }
            Direction::South => {
                if y < self.dimension - 1 {
                    Some((y + 1, x))
                } else {
                    None
                }
            }
            Direction::West => {
                if x > 0 {
                    Some((y, x - 1))
                } else {
                    None
                }
            }
        }
    }

    fn bidirectional_edge_op(
        &mut self,
        (node, direction): &Edge,
        op: impl Fn(&mut EdgeSet, Direction),
    ) -> Result<(), IndexOutOfBoundsError> {
        if let Some(neighbor) = self.neighbor((*node, *direction)) {
            if let Some(edge_set) = self.node_mut(*node) {
                op(edge_set, *direction);
                op(self.node_mut(neighbor).unwrap(), direction.inverse());
                return Ok(());
            }
        }
        Err(IndexOutOfBoundsError)
    }

    pub fn new(dimension: usize, config: &[GraphOperation]) -> Self {
        let mut graph = Self {
            dimension,
            nodes: vec![Default::default(); dimension * dimension],
            visited: vec![Default::default(); dimension * dimension],
        };

        for op in config {
            match op {
                GraphOperation::InterconnectAll => {
                    for y in 0..dimension {
                        for x in 0..dimension {
                            let edge_set = graph.node_mut((y, x)).unwrap();
                            edge_set.fill();
                            if x == 0 {
                                edge_set.remove(Direction::West);
                            }
                            if x == dimension - 1 {
                                edge_set.remove(Direction::East);
                            }
                            if y == 0 {
                                edge_set.remove(Direction::North);
                            }
                            if y == dimension - 1 {
                                edge_set.remove(Direction::South);
                            }
                        }
                    }
                }
                GraphOperation::AddBidirectional(edge) => {
                    graph
                        .bidirectional_edge_op(edge, |edge_set, direction| edge_set.add(direction))
                        .unwrap_or_else(|e| eprintln!("Warning: skipping {:?}: {:?}", op, e));
                }
                GraphOperation::RemoveBidirectional(edge) => {
                    graph
                        .bidirectional_edge_op(edge, |edge_set, direction| {
                            edge_set.remove(direction)
                        })
                        .unwrap_or_else(|e| eprintln!("Warning: skipping {:?}: {:?}", op, e));
                }
            }
        }

        graph
    }

    pub fn set_visited_at(&mut self, node: Node) -> Result<(), IndexOutOfBoundsError> {
        let idx = self.index(node).ok_or(IndexOutOfBoundsError)?;
        self.visited[idx] = true;
        Ok(())
    }

    pub fn is_visited_at(&self, node: Node) -> Result<bool, IndexOutOfBoundsError> {
        let idx = self.index(node).ok_or(IndexOutOfBoundsError)?;
        Ok(self.visited[idx])
    }
}

#[derive(Clone)]
struct Search {
    graph: SquareGraph,
    moves: Vec<Direction>,
}

impl Search {
    pub fn new(graph: SquareGraph) -> Self {
        let dim = graph.dimension();
        Self {
            graph,
            moves: vec![Direction::Nowhere; dim * dim],
        }
    }

    pub fn index(&self, node: Node) -> Option<usize> {
        self.graph.index(node)
    }

    pub fn size(&self) -> usize {
        self.graph.dimension().pow(2)
    }

    pub fn advance(&self, edge: Edge) -> Option<(Self, Node)> {
        let (node, direction) = edge;
        if let Some(edge_set) = self.graph.node(node) {
            if edge_set.contains(direction) {
                if let Some(into) = self.graph.neighbor(edge) {
                    if let Ok(false) = self.graph.is_visited_at(into) {
                        let (node, direction) = edge;

                        let mut graph = self.graph.clone();
                        graph.set_visited_at(node).unwrap();

                        let mut moves = self.moves.clone();
                        moves[self.index(node).unwrap()] = direction;

                        return Some((Self { graph, moves }, into));
                    }
                }
            }
        }
        None
    }
}

impl std::fmt::Display for Search {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..self.graph.dimension() {
            for x in 0..self.graph.dimension() {
                write!(f, "{}", {
                    let string = self.graph.node((y, x)).unwrap().to_string();
                    if let Ok(true) = self.graph.is_visited_at((y, x)) {
                        Color::Red.paint(string)
                    } else {
                        ANSIString::from(string)
                    }
                })?;
                write!(
                    f,
                    "{}",
                    match (
                        self.index((y, x)).map(|idx| self.moves[idx]),
                        self.index((y, x + 1)).map(|idx| self.moves[idx]),
                    ) {
                        (Some(Direction::East), _) | (_, Some(Direction::West)) =>
                            Color::Red.paint("─"),
                        _ => ANSIString::from(" "),
                    }
                )?;
            }
            writeln!(f)?;
            if y < self.graph.dimension() - 1 {
                for x in 0..self.graph.dimension() {
                    write!(
                        f,
                        "{}",
                        match (
                            self.index((y, x)).map(|idx| self.moves[idx]),
                            self.index((y + 1, x)).map(|idx| self.moves[idx]),
                        ) {
                            (Some(Direction::South), _) | (_, Some(Direction::North)) =>
                                Color::Red.paint("│ "),
                            _ => ANSIString::from("  "),
                        }
                    )?;
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

struct DepthFirstSearch {
    search: Option<Search>,
    from: Node,
    max_turns: usize,
    render: bool,
    previous_direction: Direction,
    solutions: Vec<Search>,
}

impl DepthFirstSearch {
    pub fn new(search: Search, from: Node, max_turns: usize, render: bool) -> Self {
        Self {
            search: Some(search),
            from,
            max_turns,
            render,
            previous_direction: Direction::Nowhere,
            solutions: Vec::new(),
        }
    }

    fn dfs(&mut self, search: Search, from: Node, turns: usize, n: usize) {
        if self.render {
            std::thread::sleep(std::time::Duration::from_millis(1000 / 60));
            print!("\x1B[2J\x1B[1;1H");
            println!("{search}");
            println!("turns = {turns}    n = {n}");
        }

        if turns > self.max_turns {
            return;
        };

        if n == search.size() {
            self.solutions.push(search);
            return;
        }

        for direction in [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ] {
            if let Some((search, node)) = search.advance((from, direction)) {
                let prev = self.previous_direction;
                self.previous_direction = direction;
                self.dfs(
                    search,
                    node,
                    turns
                        + if direction != prev && prev != Direction::Nowhere {
                            1
                        } else {
                            0
                        },
                    n + 1,
                );
            }
        }
    }

    pub fn run(mut self) -> Vec<Search> {
        let search = self.search.take().unwrap();
        self.dfs(search, self.from, 0, 1);
        self.solutions
    }
}

fn main() {
    let graph = SquareGraph::new(
        8,
        &[
            GraphOperation::InterconnectAll,
            GraphOperation::RemoveBidirectional(((7, 3), Direction::East)),
        ],
    );
    let search = Search::new(graph);
    let turns = 14;
    let dfs = DepthFirstSearch::new(search, (6, 2), turns, false);

    let routes = dfs.run();
    for route in &routes {
        println!("{route}");
    }
    if routes.is_empty() {
        println!("No route found for {turns} turns or less");
    }
}
