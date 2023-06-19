treiber_stack - A Rust Treiber Stack
====================================

A concurrent, lockless linked list - the well-known, extremely useful
[Treiber stack](https://en.wikipedia.org/wiki/Treiber_stack) data structure.

This is a handy data structure in any scenario where tasks on multiple threads
"throw something over the wall" for later processing on another thread, where
you can't afford blocking the thread in question due to contention.

To use, simply add to your `Cargo.toml`

`treiber_stack = "1.0.2"`

Check the latest version, but this library is unlikely to be updated often if ever - it
is a simple data structure.

For background on why this library exists, see
[this Reddit thread](https://www.reddit.com/r/rust/comments/13x97hb/implementing_a_trieber_stack_in_rust/).
