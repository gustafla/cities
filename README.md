# Square Graph Depth First Search

[![asciicast](https://asciinema.org/a/0JnokMBnHQCfwsxWEGP3hDbG1.svg)](https://asciinema.org/a/0JnokMBnHQCfwsxWEGP3hDbG1)

## Usage

You can run a search on a built-in graph with `cargo run`.
You can specify your own graph JSON with `--graph`.
Refer to [smallgraph.json](smallgraph.json) for the format.
Use the `--animate` (and `--delay`) option for animated output like above.
See `--help` for the other options.

## Example run:

```
Solution 1 found
Solution 2 found
Solution 3 found
Solution 4 found
Solution 5 found
Solution 6 found
Solution 7 found
Solution 8 found
Solution 9 found
Solution 10 found
Solution 11 found
Solution 12 found
Solution 13 found
Solution 14 found
Solution 15 found
Solution 16 found
Solution 17 found
Solution 18 found
With 14 turns:
┌───┬   ┬───┬───┬───┬───┬───┐   
│   │   │                   │   
├   ┼   ┼   ┼───┼   ┼───┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │       │   │   │   │   │   
└   ┴───┴───┘   └───┴   ┴───┘   

With 14 turns:
┌───┬   ┬───┬───┬───┬───┬───┐   
│   │   │                   │   
├   ┼   ┼   ┼───┼   ┼───┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│           │   │   │   │   │   
└───┴───┴───┘   └───┴   ┴───┘   

With 14 turns:
┌───┬   ┬───┬───┬───┬───┬───┐   
│   │   │                   │   
├   ┼   ┼   ┼───┼───┼───┼   ┤   
│   │   │   │           │   │   
├   ┼   ┼   ┼   ┼───┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼───┼   ┤   
│   │       │   │           │   
└   ┴───┴───┘   └───┴───┴───┘   

With 14 turns:
┌───┬   ┬───┬───┬───┬───┬───┐   
│   │   │                   │   
├   ┼   ┼   ┼───┼───┼───┼   ┤   
│   │   │   │           │   │   
├   ┼   ┼   ┼   ┼───┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼───┼   ┤   
│           │   │           │   
└───┴───┴───┘   └───┴───┴───┘   

With 14 turns:
┌───┬   ┬───┬───┬───┬   ┬───┐   
│   │   │           │   │   │   
├   ┼   ┼   ┼───┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼───┼   ┤   
│   │       │   │           │   
└   ┴───┴───┘   └───┴───┴───┘   

With 14 turns:
┌───┬   ┬───┬───┬───┬   ┬───┐   
│   │   │           │   │   │   
├   ┼   ┼   ┼───┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼───┼   ┤   
│           │   │           │   
└───┴───┴───┘   └───┴───┴───┘   

With 14 turns:
┌───┬───┬───┬───┬───┬───┬───┐   
│                           │   
├   ┼───┼   ┼───┼───┼───┼   ┤   
│   │   │   │           │   │   
├   ┼   ┼   ┼   ┼───┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼───┼   ┤   
│   │       │   │           │   
└   ┴───┴───┘   └───┴───┴───┘   

With 14 turns:
┌───┬───┬───┬───┬───┬───┬───┐   
│                           │   
├   ┼───┼   ┼───┼   ┼───┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │       │   │   │   │   │   
└   ┴───┴───┘   └───┴   ┴───┘   

With 14 turns:
┌───┬───┬───┬───┬───┬   ┬───┐   
│                   │   │   │   
├   ┼───┼   ┼───┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼───┼   ┤   
│   │       │   │           │   
└   ┴───┴───┘   └───┴───┴───┘   

With 14 turns:
┌───┬───┬───┬───┬───┬───┬───┐   
│                           │   
├   ┼───┼───┼───┼───┼───┼   ┤   
│   │                   │   │   
├   ┼   ┼───┼───┼───┼   ┼   ┤   
│   │   │           │   │   │   
├   ┼   ┼   ┼───┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼───┼   ┤   
│   │       │   │           │   
└   ┴───┴───┘   └───┴───┴───┘   

With 14 turns:
┌───┬───┬───┬───┬───┬───┬───┐   
│                           │   
├   ┼───┼───┼───┼───┼───┼   ┤   
│   │                   │   │   
├   ┼   ┼───┼───┼───┼   ┼   ┤   
│   │   │           │   │   │   
├   ┼   ┼   ┼───┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │       │   │   │   │   │   
└   ┴───┴───┘   └───┴   ┴───┘   

With 14 turns:
┌───┬───┬───┬───┬───┬───┬───┐   
│                           │   
├   ┼───┼───┼───┼───┼───┼   ┤   
│   │                   │   │   
├   ┼   ┼───┼───┼───┼   ┼   ┤   
│   │   │           │   │   │   
├   ┼   ┼   ┼───┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│           │   │   │   │   │   
└───┴───┴───┘   └───┴   ┴───┘   

With 14 turns:
┌───┬───┬───┬───┬───┬───┬───┐   
│                           │   
├   ┼───┼───┼───┼───┼───┼   ┤   
│   │                   │   │   
├   ┼   ┼───┼   ┼───┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼───┼   ┤   
│   │       │   │           │   
└   ┴───┴───┘   └───┴───┴───┘   

With 14 turns:
┌───┬───┬───┬───┬───┬───┬───┐   
│                           │   
├   ┼───┼───┼───┼   ┼───┼   ┤   
│   │           │   │   │   │   
├   ┼   ┼───┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │       │   │   │   │   │   
└   ┴───┴───┘   └───┴   ┴───┘   

With 14 turns:
┌───┬───┬───┬───┬───┬   ┬───┐   
│                   │   │   │   
├   ┼───┼───┼───┼   ┼   ┼   ┤   
│   │           │   │   │   │   
├   ┼   ┼───┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼───┼   ┤   
│   │       │   │           │   
└   ┴───┴───┘   └───┴───┴───┘   

With 14 turns:
┌───┬───┬───┬───┬───┬───┬───┐   
│                           │   
├   ┼───┼───┼───┼   ┼───┼   ┤   
│   │           │   │   │   │   
├   ┼   ┼───┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│           │   │   │   │   │   
└───┴───┴───┘   └───┴   ┴───┘   

With 14 turns:
┌───┬───┬───┬───┬───┬───┬───┐   
│                           │   
├   ┼───┼───┼───┼───┼───┼   ┤   
│   │                   │   │   
├   ┼   ┼───┼   ┼───┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼───┼   ┤   
│           │   │           │   
└───┴───┴───┘   └───┴───┴───┘   

With 14 turns:
┌───┬───┬───┬───┬───┬   ┬───┐   
│                   │   │   │   
├   ┼───┼───┼───┼   ┼   ┼   ┤   
│   │           │   │   │   │   
├   ┼   ┼───┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼   ┼   ┤   
│   │   │   │   │   │   │   │   
├   ┼   ┼   ┼   ┼   ┼───┼   ┤   
│           │   │           │   
└───┴───┴───┘   └───┴───┴───┘   
```
