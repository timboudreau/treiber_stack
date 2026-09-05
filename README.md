treiber_stack - A Rust Treiber Stack
====================================

A concurrent, lockless linked list - the well-known, extremely useful
[Treiber stack](https://en.wikipedia.org/wiki/Treiber_stack) data structure.

This is a handy data structure in any scenario where tasks on multiple threads
"throw something over the wall" for later processing on another thread, where
you can't afford blocking the thread in question due to contention.

To use, simply add to your `Cargo.toml`
`treiber_stack = "1.2.0"`


What's Here?
------------

This crate contains two types implementing the treiber-stack structure - the original, arc-swap-based
one introduced in version 1.0 (available with the feature-flag `stack` which is on by default), 
and a simpler opaque queue implementation introduced in 1.2, enabled with the feature-flags 
`queue` or `queue-stats`.

Differences between the two:

 * `TreiberStack` exposes a *lot* of internal state (note that the internal state of a Treiber stack
    is necessarily ephemeral - there is no guarantee that the result of a call still reflects the
    internal state of the stack), and wraps its contents in `Arc`s to allow visibility of its
    contents.
 * `TreiberQueue` is a simple, opaque queue.  When you push an item in, the queue takes ownership
    of it (in fact, using `Box::leak` on it under the hood, and `*Box::from_raw()` to recover it
    on pop or drop it when the queue is dropped), and it is inaccessible until popped.  Implementation-wise,
    this is considerably simpler and has less overhead.  Unlike `TreiberStack`, the queue *does* use
    `unsafe` code, since it deals in pointers.

Unless you *need* either the exposed state of `TreiberStack` or have a requirement to avoid directly
depending on libraries using `unsafe`, `TreiberQueue` is likely to be the more performant choice.

Feature Flags
-------------

 * `stack` - on-by-default; includes the `TreiberStack` type.
 * `queue` - off-by-default; includes the `TreiberQueue` type.
 * `queue-stats` - off-by-default; enables the `queue` feature; adds a `stats()` method to `TreiberQueue`
    that can report stats of the number of elements leaked and unleaked (pushed and popped) by a
    queue.
 * `queue-read-relaxed` - off-by-default; causes `TreiberQueue` to use `std::sync::ordering::Relaxed`
    for the read-portion of a push or pop operation.  Note that writes *always* use `std::sync::ordering::SeqCst`
    (not doing so can result in double-freeing), so every push or pop operation involves a sequentially
    consistent operation.  The default ordering without this flag is sequentially consistent for both
    read and write operations.

### License

This code is licensed under the MIT license.
