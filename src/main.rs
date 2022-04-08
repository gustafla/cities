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
pub struct SquareGraph {
    dimension: usize,
    nodes: Vec<EdgeSet>,
}

impl SquareGraph {
    pub fn index(&self, node: Node) -> Option<usize> {
        if node.0 < self.dimension && node.1 < self.dimension {
            Some(node.0 * self.dimension + node.1)
        } else {
            None
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
}

#[derive(Clone)]
struct Search<'a> {
    graph: &'a SquareGraph,
    visited: Vec<bool>,
    moves: Vec<Option<Direction>>,
}

impl<'a> Search<'a> {
    pub fn new(graph: &'a SquareGraph) -> Self {
        let dim = graph.dimension();
        Self {
            graph,
            visited: vec![false; dim * dim],
            moves: vec![None; dim * dim],
        }
    }

    pub fn index(&self, node: Node) -> Option<usize> {
        self.graph.index(node)
    }

    fn update_visited(&self, node: Node) -> Result<Vec<bool>, IndexOutOfBoundsError> {
        let idx = self.index(node).ok_or(IndexOutOfBoundsError)?;
        let mut visited = self.visited.clone();
        visited[idx] = true;
        Ok(visited)
    }

    fn has_visited_at(&self, node: Node) -> Result<bool, IndexOutOfBoundsError> {
        let idx = self.index(node).ok_or(IndexOutOfBoundsError)?;
        Ok(self.visited[idx])
    }

    pub fn depth(&self) -> usize {
        self.graph.dimension().pow(2)
    }

    pub fn advance(&self, edge: Edge) -> Option<(Self, Node)> {
        let (node, direction) = edge;
        if let Some(edge_set) = self.graph.node(node) {
            if edge_set.contains(direction) {
                if let Some(into) = self.graph.neighbor(edge) {
                    if let Ok(false) = self.has_visited_at(into) {
                        let (node, direction) = edge;

                        let visited = self.update_visited(node).unwrap();
                        let mut moves = self.moves.clone();
                        moves[self.index(node).unwrap()] = Some(direction);

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
        for y in 0..self.graph.dimension() {
            for x in 0..self.graph.dimension() {
                write!(f, "{}", self.graph.node((y, x)).unwrap())?;
                write!(
                    f,
                    "{}",
                    match (
                        self.index((y, x)).and_then(|idx| self.moves[idx]),
                        self.index((y, x + 1)).and_then(|idx| self.moves[idx]),
                    ) {
                        (Some(Direction::East), _) | (_, Some(Direction::West)) => "───",
                        _ => "   ",
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
                            self.index((y, x)).and_then(|idx| self.moves[idx]),
                            self.index((y + 1, x)).and_then(|idx| self.moves[idx]),
                        ) {
                            (Some(Direction::South), _) | (_, Some(Direction::North)) => "│   ",
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
            search: Some(Search::new(&graph)),
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

        for dir in [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ] {
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
        cli_start.unwrap_or(start.unwrap_or_default()),
        turns,
        opts.animate.then(|| opts.delay),
        opts.first,
    );

    let routes = dfs.run();
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
