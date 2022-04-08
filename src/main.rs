use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub enum Direction {
    North = 0b0001,
    East = 0b0010,
    South = 0b0100,
    West = 0b1000,
}

impl Direction {
    pub fn inverse(&self) -> Self {
        match self {
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

#[derive(Debug, Deserialize)]
pub enum GraphOperation {
    InterconnectAll,
    AddBidirectional(Edge),
    RemoveBidirectional(Edge),
}

#[derive(Debug)]
pub struct IndexOutOfBoundsError;

#[derive(Clone)]
pub struct Square<T: Default + Clone> {
    size: usize,
    buf: Vec<T>,
}

impl<T: Default + Clone> Square<T> {
    fn new(size: usize) -> Self {
        Self {
            size,
            buf: vec![Default::default(); size * size],
        }
    }

    fn size(&self) -> usize {
        self.size
    }

    pub fn index(&self, node: Node) -> Option<usize> {
        if node.0 < self.size && node.1 < self.size {
            Some(node.0 * self.size + node.1)
        } else {
            None
        }
    }

    pub fn get(&self, node: Node) -> Option<&T> {
        self.index(node).and_then(|idx| self.buf.get(idx))
    }

    pub fn get_mut(&mut self, node: Node) -> Option<&mut T> {
        self.index(node).and_then(|idx| self.buf.get_mut(idx))
    }

    pub fn set(&mut self, node: Node, value: T) -> Result<(), IndexOutOfBoundsError> {
        if let Some(idx) = self.index(node) {
            self.buf[idx] = value;
            Ok(())
        } else {
            Err(IndexOutOfBoundsError)
        }
    }

    pub fn neighbor(&self, ((y, x), dir): Edge) -> Option<Node> {
        use Direction::*;
        match dir {
            North => {
                if y > 0 {
                    Some((y - 1, x))
                } else {
                    None
                }
            }
            East => {
                if x < self.size - 1 {
                    Some((y, x + 1))
                } else {
                    None
                }
            }
            South => {
                if y < self.size - 1 {
                    Some((y + 1, x))
                } else {
                    None
                }
            }
            West => {
                if x > 0 {
                    Some((y, x - 1))
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct SquareGraph {
    nodes: Square<EdgeSet>,
}

impl SquareGraph {
    fn bidirectional_edge_op(
        &mut self,
        (node, direction): &Edge,
        op: impl Fn(&mut EdgeSet, Direction),
    ) -> Result<(), IndexOutOfBoundsError> {
        if let Some(neighbor) = self.nodes.neighbor((*node, *direction)) {
            if let Some(edge_set) = self.nodes.get_mut(*node) {
                op(edge_set, *direction);
                op(self.nodes.get_mut(neighbor).unwrap(), direction.inverse());
                return Ok(());
            }
        }
        Err(IndexOutOfBoundsError)
    }

    pub fn new(size: usize, config: &[GraphOperation]) -> Self {
        let mut graph = Self {
            nodes: Square::new(size),
        };

        for op in config {
            match op {
                GraphOperation::InterconnectAll => {
                    for y in 0..size {
                        for x in 0..size {
                            let edge_set = graph.nodes.get_mut((y, x)).unwrap();
                            edge_set.fill();
                            if x == 0 {
                                edge_set.remove(Direction::West);
                            }
                            if x == size - 1 {
                                edge_set.remove(Direction::East);
                            }
                            if y == 0 {
                                edge_set.remove(Direction::North);
                            }
                            if y == size - 1 {
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

    pub fn size(&self) -> usize {
        self.nodes.size()
    }

    pub fn nodes(&self) -> &Square<EdgeSet> {
        &self.nodes
    }
}

#[derive(Clone)]
struct Search<'a> {
    graph: &'a SquareGraph,
    visited: Square<bool>,
    moves: Square<Option<Direction>>,
}

impl<'a> Search<'a> {
    pub fn new(graph: &'a SquareGraph) -> Self {
        let size = graph.size();
        Self {
            graph,
            visited: Square::new(size),
            moves: Square::new(size),
        }
    }

    fn update_visited(&self, node: Node) -> Result<Square<bool>, IndexOutOfBoundsError> {
        let mut visited = self.visited.clone();
        visited.set(node, true)?;
        Ok(visited)
    }

    pub fn depth(&self) -> usize {
        self.graph.size().pow(2)
    }

    pub fn advance(&self, edge: Edge) -> Option<(Self, Node)> {
        let (node, direction) = edge;
        if let Some(edge_set) = self.graph.nodes().get(node) {
            if edge_set.contains(direction) {
                if let Some(into) = self.graph.nodes().neighbor(edge) {
                    if let Some(false) = self.visited.get(into) {
                        let visited = self.update_visited(node).unwrap();
                        let mut moves = self.moves.clone();
                        moves.set(node, Some(direction)).unwrap();

                        return Some((
                            Self {
                                graph: self.graph,
                                visited,
                                moves,
                            },
                            into,
                        ));
                    }
                }
            }
        }
        None
    }
}

impl<'a> std::fmt::Display for Search<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Direction::*;

        for y in 0..self.graph.size() {
            for x in 0..self.graph.size() {
                write!(f, "{}", self.graph.nodes().get((y, x)).unwrap())?;
                write!(
                    f,
                    "{}",
                    match (self.moves.get((y, x)).unwrap(), self.moves.get((y, x + 1))) {
                        (Some(East), _) | (_, Some(Some(West))) => "───",
                        _ => "   ",
                    }
                )?;
            }
            writeln!(f)?;
            if y < self.graph.size() - 1 {
                for x in 0..self.graph.size() {
                    write!(
                        f,
                        "{}",
                        match (self.moves.get((y, x)).unwrap(), self.moves.get((y + 1, x))) {
                            (Some(South), _) | (_, Some(Some(North))) => "│   ",
                            _ => "    ",
                        }
                    )?;
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

struct DepthFirstSearch<'a> {
    search: Option<Search<'a>>,
    from: Node,
    max_turns: Option<usize>,
    animate: Option<u64>,
    first: bool,
    step: usize,
    solutions: Vec<(Search<'a>, usize)>,
}

impl<'a> DepthFirstSearch<'a> {
    pub fn new(
        graph: &'a SquareGraph,
        from: Node,
        max_turns: Option<usize>,
        animate: Option<u64>,
        first: bool,
    ) -> Self {
        Self {
            search: Some(Search::new(graph)),
            from,
            max_turns,
            animate,
            first,
            step: 0,
            solutions: Vec::new(),
        }
    }

    fn dfs(
        &mut self,
        search: Search<'a>,
        from: Node,
        prev_dir: Option<Direction>,
        turns: usize,
        depth: usize,
    ) {
        use Direction::*;

        if let Some(max) = self.max_turns {
            if turns > max {
                return;
            };
        }

        self.step += 1;

        if let Some(delay) = self.animate {
            std::thread::sleep(std::time::Duration::from_millis(delay));
            print!("\x1B[2J\x1B[1;1H");
            println!("{search}");
            println!("step = {}    turns = {turns}    depth = {depth}", self.step);
        }

        if depth == search.depth() {
            self.solutions.push((search, turns));
            if self.animate.is_none() {
                println!("Solution {} found", self.solutions.len());
            }
            return;
        }

        for dir in [North, East, South, West] {
            if let Some((search, node)) = search.advance((from, dir)) {
                self.dfs(
                    search,
                    node,
                    Some(dir),
                    turns + prev_dir.and_then(|p| (p != dir).then(|| 1)).unwrap_or(0),
                    depth + 1,
                );
                if self.first && !self.solutions.is_empty() {
                    return;
                }
            }
        }
    }

    pub fn run(mut self) -> Vec<(Search<'a>, usize)> {
        let search = self.search.take().unwrap();
        self.dfs(search, self.from, None, 0, 1);
        self.solutions
    }
}

#[derive(Deserialize)]
struct Problem {
    size: usize,
    start: Option<Node>,
    max_turns: Option<usize>,
    graph_ops: Vec<GraphOperation>,
}

impl Default for Problem {
    fn default() -> Self {
        Self::new(8)
            .start_from((6, 2))
            .max_turns(14)
            .operation(GraphOperation::InterconnectAll)
            .operation(GraphOperation::RemoveBidirectional((
                (7, 3),
                Direction::East,
            )))
    }
}

impl Problem {
    fn new(size: usize) -> Self {
        Self {
            size,
            start: None,
            max_turns: None,
            graph_ops: Vec::new(),
        }
    }

    fn operation(mut self, op: GraphOperation) -> Self {
        self.graph_ops.push(op);
        self
    }

    fn start_from(mut self, node: Node) -> Self {
        self.start = Some(node);
        self
    }

    fn max_turns(mut self, turns: usize) -> Self {
        self.max_turns = Some(turns);
        self
    }

    fn build(self) -> (SquareGraph, Option<Node>, Option<usize>) {
        (
            SquareGraph::new(self.size, &self.graph_ops),
            self.start,
            self.max_turns,
        )
    }
}

#[derive(Parser)]
struct Cli {
    /// Output an animation of the search progress
    #[clap(short, long)]
    animate: bool,
    /// Animation frame delay in milliseconds
    #[clap(short, long, default_value_t = 33, requires = "animate")]
    delay: u64,
    /// Stop after one solution
    #[clap(short, long)]
    first: bool,
    /// Load graph from json file instead of using built-in
    #[clap(short, long)]
    graph: Option<PathBuf>,
    /// Maximum number of direction changes allowed during route
    #[clap(short, long)]
    max_turns: Option<usize>,
    /// Starting point y-coordinate on the grid
    #[clap(requires = "starting-point-x")]
    starting_point_y: Option<usize>,
    /// Starting point x-coordinate on the grid
    starting_point_x: Option<usize>,
}

fn main() -> Result<()> {
    let opts = Cli::parse();

    // Parse problem
    let (graph, start, turns) = match &opts.graph {
        None => Problem::default(),
        Some(path) => {
            let file =
                std::fs::File::open(path).context(format!("Failed to open {}", path.display()))?;
            serde_json::de::from_reader(file)
                .context(format!("Failed to parse {}", path.display()))?
        }
    }
    .build();

    // Convert cli options to Node-tuple
    let cli_start = match (opts.starting_point_y, opts.starting_point_x) {
        (Some(y), Some(x)) => Some((y, x)),
        _ => None,
    };

    let turns = match (opts.max_turns, turns) {
        // Prefer cli option max_turns
        (t @ Some(_), _) => t,
        // But use Problem's max_turns if not present
        (None, t @ Some(_)) => t,
        // Or just don't limit at all
        _ => None,
    };

    let dfs = DepthFirstSearch::new(
        &graph,
        cli_start.unwrap_or_else(|| start.unwrap_or_default()),
        turns,
        opts.animate.then(|| opts.delay),
        opts.first,
    );

    let mut routes = dfs.run();
    routes.sort_unstable_by_key(|e| std::cmp::Reverse(e.1));
    for (route, turns) in &routes {
        println!("With {turns} turns:");
        println!("{route}");
    }
    if routes.is_empty() {
        println!(
            "No route found{}",
            if let Some(turns) = turns {
                format!(" for {turns} turns or less")
            } else {
                "".into()
            }
        );
    }
    Ok(())
}
