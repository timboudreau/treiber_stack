/*
 * The MIT License
 *
 * Copyright 2023-2026, Tim Boudreau.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
//! Provides a simplified, `Arc`-free Treiber stack - possibly the simplest
//! form of one possible in Rust - an opaque queue which items can be pushed
use std::{
    marker::PhantomData,
    sync::atomic::{
        AtomicUsize,
        Ordering::{Relaxed, SeqCst},
    },
};

// Pending: add an option, requiring the nightly compiler, to use the native
// AtomicU128 on platforms that support it.

type AtomicU128 = portable_atomic::AtomicU128;
const CLEANUP_GENERATION_CELLS: usize = 16;

#[cfg(not(feature = "queue-read-relaxed"))]
mod ord {
    use std::sync::atomic::Ordering;
    pub(super) const READ_ORDERING: Ordering = Ordering::SeqCst;
    /// Under all scenarios we need the *write* order to be `SeqCst` or
    /// it is possible to double-free.
    pub(super) const WRITE_ORDERING: Ordering = Ordering::SeqCst;
}

#[cfg(feature = "queue-read-relaxed")]
mod ord {
    use std::sync::atomic::Ordering;
    pub(super) const READ_ORDERING: Ordering = Ordering::Relaxed;
    /// Under all scenarios we need the *write* order to be `SeqCst` or
    /// it is possible to double-free.
    pub(super) const WRITE_ORDERING: Ordering = Ordering::SeqCst;
}

/// A simplified treiber stack with less overhead than the arc-swap-based one.
/// A lockless, thread-safe, opaque LIFO queue implemented over `AtomicU128`.
///
/// Included with the feature-flags `queue` or `queue-stats`.
///
/// Internally, it `Box::leak`s values that are pushed, and unboxes them when
/// popped. Since anything pushed onto the queue effectively disappears from the
/// universe of the rest of the application until it is popped, and is not observable
/// or borrowable, this bypasses needing `Arc` or other smart-pointer types to
/// manage the lifecycle of queue members.
///
/// Half the bits of each cell store the leaked value pointer; the other half stores
/// either null for the tail cell, or a pointer to the next cell.  The only point
/// of mutation is the single `AtomicU128` that is its state.
///
/// Dropping a `TreiberQueue` will cause all of its contents to be drained and dropped.
///
/// What can be known about the internal state of the queue is whether it is empty,
/// and, when pushing, if it had been empty prior to that push.  Information about
/// the state of the queue is, necessarily, ephemeral and always potentially-out-of-date.
/// `is_empty()` is provided as a convenience for the use case of thread pools that
/// want to completely drain pending work that may be arriving concurrently.
///
#[derive(Debug)]
pub struct TreiberQueue<T: Sized + 'static> {
    cell: HeadCell,
    cleanup: Cleanup<CLEANUP_GENERATION_CELLS>,
    _pd: PhantomData<T>,
}

// This is just a thin wrapper on the underlying `HeadCell` that imposes a consistent
// type, to keep the front-facing API separate from the implementation.
impl<T: Sized + 'static> TreiberQueue<T> {
    /// Create a new queue.
    pub const fn new() -> Self {
        Self {
            cell: HeadCell::new(),
            cleanup: Cleanup::new(),
            _pd: PhantomData,
        }
    }

    #[cfg(any(feature = "queue-stats", test))]
    /// Some diagnostic stats - the number of items currently pushed and popped and structs
    /// (both values and head-cells) leaked and unleaked.
    ///
    /// Cells unleaked will lag the number of pops and will be double the number
    /// of pushes, minus one for each time the cell was pushed to when it was empty
    /// (the initial cell head does not need to leak an adjacent head-cell since there
    /// isn't one) - unleaking cells happens every `n` cells (where `n` is
    /// implementation-determined) in generations to avoid double-freeing head cells.
    ///
    /// If this stack is empty, pushes should equal pops.  Stats are per-queue,
    /// not global, and only collected if the feature `queue-stats` is enabled.
    pub fn stats(&self) -> (usize, usize, usize, usize) {
        self.cell.stats()
    }

    #[cfg(any(feature = "queue-stats", test))]
    /// Returns a best-effort tally of the number of pushes minus the number of pops
    /// this queue has seen. The result should be treated as a hint, not ground-truth
    /// (as should all reports of the internal state of a lockless structure).
    pub fn approximate_len(&self) -> usize {
        self.cell.approximate_len()
    }

    /// Push an item onto the queue. Returns true if the queue was empty prior to this
    /// call - i.e. the caller's pushed item became the tail item in the stack (note
    /// that, this being a lockless, concurrent queue, that **does not mean** that it
    /// is *still* the head of the stack by the time the caller receives and reads
    /// the return `bool`).
    ///
    /// ```
    /// let q = treiber_stack::TreiberQueue::<usize>::new();
    /// assert!(q.is_empty());
    /// q.push(23);
    /// assert!(!q.is_empty());
    /// let n = q.pop();
    /// assert_eq!(n, Some(23));
    /// assert!(q.is_empty());
    /// ```
    #[inline(always)]
    pub fn push(&self, t: T) -> bool {
        self.cell.push::<T>(t)
    }

    /// Convenience method to push an item onto the queue that the
    /// caller already has boxed, such as boxed dyn functions.
    #[inline(always)]
    pub fn push_boxed(&self, t: Box<T>) -> bool {
        self.cell.push_boxed::<T>(t)
    }

    /// Pop an item off of the queue.
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        if let Some((item, inner)) = self.cell.pop::<T>() {
            self.cleanup.push(inner, &self.cell);
            Some(item)
        } else {
            None
        }
    }

    /// Determine if the queue is currently empty (which may not be true a nanosecond
    /// after this method returns); useful to spin a thread pulling work from a queue
    /// to avoid liveness issues.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.cell.is_empty()
    }

    /// Empty the entire contents of the queue.  The result will be in LIFO order.
    /// ```
    /// let q = treiber_stack::TreiberQueue::<usize>::new();
    /// (0_usize..5).for_each(|ix| assert_eq!(ix == 0, q.push(ix)));
    /// assert!(!q.is_empty());
    /// let drained = q.drain();
    /// assert!(q.is_empty());
    /// assert_eq!(drained, vec![4, 3, 2, 1, 0]);
    /// ```
    pub fn drain(&self) -> Vec<T> {
        self.cell.drain::<_, T>(&self.cleanup)
    }

    /// Drain up to `n_items` items from the queue.
    pub fn drain_no_more_than(&self, n_items: usize) -> Vec<T> {
        self.cell.drain_no_more_than::<_, T>(n_items, &self.cleanup)
    }
}

/// Ensures all of the queue contents are dropped.
impl<T: Sized + 'static> Drop for TreiberQueue<T> {
    fn drop(&mut self) {
        // ensure the contents are dropped, using `drain()` which iterates rather
        // than recurses, avoiding a stack-overflow if dropping a very large queue.
        while !self.is_empty() {
            let _ = self.drain();
        }
    }
}

impl<T: Sized + 'static> Default for TreiberQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a `TreiberQueue` with a single initial element.
///
/// ```
/// let q = treiber_stack::TreiberQueue::from(42_usize);
/// assert!(!q.is_empty());
/// assert_eq!(Some(42), q.pop());
/// assert!(q.is_empty());
/// ```
impl<T: Sized + 'static> From<T> for TreiberQueue<T> {
    fn from(value: T) -> Self {
        let result = Self::new();
        result.push(value);
        result
    }
}

/// Internal implementation of the queue; deliberately untyped.
#[derive(Default, Debug)]
struct HeadCell {
    state: AtomicU128,
    #[cfg(any(feature = "queue-stats", test))]
    leak_counter: AtomicUsize,
    #[cfg(any(feature = "queue-stats", test))]
    unleak_counter: AtomicUsize,
    #[cfg(any(feature = "queue-stats", test))]
    pop_counter: AtomicUsize,
    #[cfg(any(feature = "queue-stats", test))]
    push_counter: AtomicUsize,
}

impl HeadCell {
    const fn new() -> Self {
        HeadCell {
            state: AtomicU128::new(0),
            #[cfg(any(feature = "queue-stats", test))]
            leak_counter: AtomicUsize::new(0),
            #[cfg(any(feature = "queue-stats", test))]
            unleak_counter: AtomicUsize::new(0),
            #[cfg(any(feature = "queue-stats", test))]
            pop_counter: AtomicUsize::new(0),
            #[cfg(any(feature = "queue-stats", test))]
            push_counter: AtomicUsize::new(0),
        }
    }

    /// Returns pushes, pops, leaks, unleaks
    #[cfg(any(feature = "queue-stats", test))]
    fn stats(&self) -> (usize, usize, usize, usize) {
        (
            self.push_counter.load(Relaxed),
            self.pop_counter.load(Relaxed),
            self.leak_counter.load(Relaxed),
            self.unleak_counter.load(Relaxed),
        )
    }

    #[cfg(any(feature = "queue-stats", test))]
    fn approximate_len(&self) -> usize {
        let pushes = self.leak_counter.load(Relaxed);
        let pops = self.pop_counter.load(Relaxed);
        if pops > pushes { 0 } else { pushes - pops }
    }

    #[inline(always)]
    fn unleak<T: Sized + 'static>(&self, p: *mut T) -> T {
        let bx = unsafe { Box::from_raw(p) };
        #[cfg(any(feature = "queue-stats", test))]
        self.unleak_counter.fetch_add(1, Relaxed);
        *bx
    }

    #[inline(always)]
    fn leak<T: Sized + 'static>(&self, p: T) -> *mut T {
        self.leak_boxed::<T>(Box::new(p))
    }

    #[inline(always)]
    fn leak_boxed<T: Sized + 'static>(&self, p: Box<T>) -> *mut T {
        #[cfg(any(feature = "queue-stats", test))]
        self.leak_counter.fetch_add(1, Relaxed);
        Box::leak(p)
    }

    /// Remove the last inserted item from the queue.  Returns the popped item, and if present,
    /// the address of the head that formerly contained it so it can be unleaked.
    fn pop<T: Sized + 'static>(&self) -> Option<(T, usize)> {
        let res = self
            .state
            .fetch_update(ord::WRITE_ORDERING, ord::READ_ORDERING, |old| {
                // This closure may be called multiple times, so we do not unleak anything in here.
                // Complete information about the old head cell will be returned
                let cell = LinkCell(old);
                let n = cell.next_cell_value();
                Some(n)
            });
        match res {
            Ok(old) => {
                let cell = LinkCell(old);
                if let Some(value) = cell.value::<T>() {
                    let unleaked = self.unleak(value);
                    #[cfg(any(feature = "queue-stats", test))]
                    self.pop_counter.fetch_add(1, Relaxed);
                    return Some((unleaked, cell.inner_bits()));
                } else {
                    return None;
                }
            }
            Err(_curr) => {
                // assuming it's guaranteed that `fetch_update` will never return an error,
                // we could simply use unreachable!() here.
                None
            }
        }
    }

    #[inline(always)]
    fn push<T: Sized + 'static>(&self, val: T) -> bool {
        self.push_boxed::<T>(Box::new(val))
    }

    /// Returns true if the cell was empty prior to this call.
    fn push_boxed<T: Sized + 'static>(&self, val: Box<T>) -> bool {
        // Okay, we immediately leak the value
        let leaked = self.leak_boxed(val);
        // And make a cell with it
        let mut new_head = LinkCell::new_ptr(leaked);

        debug_assert!(new_head.value_bits() != 0, "new head has no value bits");

        // We leak that; we will use the fact that we have a mutable reference to it
        // to update it each time the closure is called until it succeeds
        // let new_head_leaked = leak(new_head);
        let mut last_leaked: Option<*mut LinkCell> = None;
        let res = self
            .state
            .fetch_update(ord::WRITE_ORDERING, ord::READ_ORDERING, |old| {
                if let Some(prev) = last_leaked.take() {
                    let _ = self.unleak(prev);
                    new_head.set_inner(0);
                }
                if old == 0 {
                    return Some(new_head.0);
                }

                debug_assert!(new_head.value_bits() != 0, "Value is unset");
                let old = LinkCell(old);

                // This is now going to be stored in the lower half of the u128, so leak it.
                let old = self.leak(old);
                last_leaked = Some(old);

                new_head.set_inner_ptr(old);

                debug_assert!(new_head.inner_bits() != 0, "Inner bits changed nothing");
                debug_assert!(new_head.value_bits() != 0, "Value bits got cleared");

                Some(new_head.0)
            });
        match res {
            Ok(old) => {
                #[cfg(any(feature = "queue-stats", test))]
                self.push_counter.fetch_add(1, Relaxed);
                return old == 0;
            }
            Err(e) => {
                unreachable!("Err in push: {:?}", e);
            }
        }
    }

    /// Determine if the queue is empty at the instant this method was called.
    fn is_empty(&self) -> bool {
        self.state.load(Relaxed) == 0
    }

    /// Empty the entire contents of the queue.  The result will be in LIFO order.
    fn drain<const N: usize, T: Sized + 'static>(&self, cleanup: &Cleanup<N>) -> Vec<T> {
        let mut result = Vec::<T>::new();
        while !self.is_empty() {
            if let Some((item, inner)) = self.pop() {
                result.push(item);
                cleanup.push(inner, self);
            }
        }
        result
    }

    fn drain_no_more_than<const N: usize, T: Sized + 'static>(
        &self,
        n_items: usize,
        cleanup: &Cleanup<N>,
    ) -> Vec<T> {
        let mut result = Vec::<T>::new();
        while !self.is_empty() {
            if let Some((item, inner)) = self.pop() {
                result.push(item);
                cleanup.push(inner, self);
                if result.len() == n_items {
                    break;
                }
            }
        }
        result
    }
}

const MASK_UPPER: u128 = ((u64::MAX as u128) << 64) as u128;
const MASK_LOWER: u128 = u64::MAX as u128;

#[derive(Debug, Eq, PartialEq)]
struct LinkCell(u128);

/// This is both the internal implementation of the logic to carve pointers into a u128,
/// and the type which is stored in the right hand side of the head cell to point to
/// the next cell.
impl LinkCell {
    const fn new(val: usize) -> Self {
        Self((val as u128) << 64)
    }

    fn new_ptr<T: Sized + 'static>(value: *mut T) -> Self {
        let val = value.addr() as usize;
        Self::new(val)
    }

    #[inline(always)]
    const fn value_bits(&self) -> usize {
        (self.0 >> 64) as usize
    }

    #[inline(always)]
    const fn inner_bits(&self) -> usize {
        (self.0 & MASK_LOWER) as usize
    }

    #[inline(always)]
    const fn set_inner(&mut self, inner: usize) {
        let masked = self.0 & MASK_UPPER; // preserved the value bits
        debug_assert!(masked != 0, "Cell should not be empty");
        self.0 = masked | inner as u128;
        debug_assert!(self.inner_bits() == inner, "Inner bits cleared?");
        debug_assert!(self.value_bits() != 0, "Value bits cleared?");
    }

    #[inline(always)]
    fn set_inner_ptr(&mut self, inner: *mut Self) {
        self.set_inner(inner.addr());
    }

    #[allow(dead_code)]
    const fn inner<T: Sized + 'static>(&self) -> Option<*mut T> {
        let addr = (self.0 & MASK_LOWER) as usize;
        if addr != 0 {
            let ptr = std::ptr::with_exposed_provenance_mut::<T>(addr);
            Some(ptr)
        } else {
            None
        }
    }

    #[inline(always)]
    const fn value<T: Sized + 'static>(&self) -> Option<*mut T> {
        let addr = ((self.0 & MASK_UPPER) >> 64) as usize;
        if addr != 0 {
            let ptr = std::ptr::with_exposed_provenance_mut::<T>(addr);
            Some(ptr)
        } else {
            None
        }
    }

    #[inline(always)]
    const fn next_cell_value(&self) -> u128 {
        let next_addr = (self.0 & MASK_LOWER) as usize;
        if next_addr != 0 {
            let new_head = std::ptr::with_exposed_provenance_mut::<Self>(next_addr);
            let borr = unsafe { &*(new_head as *const LinkCell) };
            borr.0
        } else {
            0
        }
    }
}

/// We need to delay dropping head-cells an coalesce the addresses, as two operations
/// can (and rarely do) result in attempting to drop the same head cell twice.
struct Cleanup<const N: usize> {
    a_set: [AtomicUsize; N],
    b_set: [AtomicUsize; N],
    cursor: AtomicUsize,
}

impl<const N: usize> std::fmt::Debug for Cleanup<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pos = self.cursor.load(Relaxed) % N;
        f.debug_struct("Cleanup")
            .field("cursor", &self.cursor)
            .field("position", &pos)
            .finish()
    }
}

impl<const N: usize> Cleanup<N> {
    const fn new() -> Self {
        let a_set: [AtomicUsize; N] = [const { AtomicUsize::new(0) }; N];
        let b_set: [AtomicUsize; N] = [const { AtomicUsize::new(0) }; N];
        let cursor: AtomicUsize = AtomicUsize::new(0);
        Self {
            a_set,
            b_set,
            cursor,
        }
    }

    #[inline]
    fn push(&self, ptr: usize, owner: &HeadCell) {
        if ptr == 0 {
            // Null pointer; do nothing.  We check this here rather than in pop()
            // to ensure it is always done by doing it only in one place.
            return;
        }
        // We keep the cursor incrementing monotonically
        let pos = self
            .cursor
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // Position of the cursor relative to the start of the target buffer
        let offset = pos % N;
        // Which buffer to use, so we are always visiting items old enough to contain both copies
        // of the same address from a concurrent pop.
        let generation = pos / N;
        let (target, other) = if generation.is_multiple_of(2) {
            (&self.a_set, &self.b_set)
        } else {
            (&self.b_set, &self.a_set)
        };
        // Take the old value, in case it is non-zero
        let old = target[offset].swap(ptr, std::sync::atomic::Ordering::SeqCst);

        let run_cleanup = old != 0 || (offset == 0 && generation > 0);

        if run_cleanup {
            let _cleaned = Self::cleanup(other, old, owner);
        } else {
            Self::cleanup_one(old, owner);
        }
    }

    fn cleanup(arr: &[AtomicUsize; N], prev: usize, owner: &HeadCell) -> usize {
        // Using a set solves prunes duplicates generated by concurrent pops:
        let mut set = std::collections::BTreeSet::new();
        if prev != 0 {
            set.insert(prev);
        }
        let mut encountered = 0_usize;
        for item in arr.iter() {
            let val = item.swap(0, SeqCst);
            if val != 0 {
                encountered += 1;
                set.insert(val);
            }
        }
        if set.len() < encountered {
            println!("ENCOUNTERED DUPS: {}", encountered - set.len());
        }
        let mut cleaned_up = 0;
        for addr in set {
            if Self::cleanup_one(addr, owner) {
                cleaned_up += 1;
            }
        }
        cleaned_up
    }

    fn cleanup_one(addr: usize, owner: &HeadCell) -> bool {
        if addr == 0 {
            return false;
        }
        let pt = std::ptr::with_exposed_provenance_mut::<LinkCell>(addr);
        owner.unleak(pt);
        true
    }

    // used only in drop, where we don't have a reference to the owner
    fn unleak(p: *mut LinkCell) -> LinkCell {
        let bx = unsafe { Box::from_raw(p) };
        *bx
    }
}

impl<const N: usize> Drop for Cleanup<N> {
    fn drop(&mut self) {
        for arr in [&self.a_set, &self.b_set] {
            for p in arr.iter() {
                let addr = p.swap(0, SeqCst);
                if addr != 0 {
                    let pt = std::ptr::with_exposed_provenance_mut::<LinkCell>(addr);
                    Self::unleak(pt);
                }
            }
        }
    }
}

#[cfg(all(feature = "queue", test))]
mod queue_tests {
    use super::*;
    use portable_atomic::AtomicUsize;
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering::*},
        },
        time::{Duration, Instant, SystemTime},
    };

    static CLEANUP: Cleanup<16> = Cleanup::new();

    #[test]
    fn test_head_cell() {
        const N_ITEMS: usize = 10;
        let cell = HeadCell::default();
        let mut items = Vec::new();

        for i in 0..N_ITEMS {
            let t = Thing::new(i + 1);
            items.push(t.clone());
            cell.push(t);
        }
        assert!(!cell.is_empty());
        items.reverse();

        let (pushes, pops, leaks, unleaks) = cell.stats();
        assert_eq!(leaks, 19, "Wrong number of leaks before drain");
        assert_eq!(pushes, 10, "Wrong number of pushes before drain");
        assert_eq!(pops, 0, "Wrong number of pops before drain");
        assert_eq!(unleaks, 0, "Wrong number of unleaks before drain");
        let contents = cell.drain::<_, Thing>(&CLEANUP);
        let (pushes, pops, leaks, unleaks) = cell.stats();
        assert_eq!(leaks, 19, "Wrong number of leaks after drain");
        assert_eq!(pushes, pops, "Pushes does not equal pops after drain");
        assert_eq!(10, unleaks, "Wrong number of unleaks after drain");
        assert!(cell.is_empty());
        assert_eq!(items, contents);
    }

    #[test]
    fn test_concurrent_pushes_then_concurrent_pops() {
        const N_PUSH_THREADS: usize = 7;
        const N_POP_THREADS: usize = 4;
        const N_ITEMS: usize = 500_000;
        let push_start_latch = TestThreadLatch::<N_PUSH_THREADS>::new();
        let q = Arc::new(TreiberQueue::<MultithreadQueueItem>::new());

        let mut push_threads = Vec::with_capacity(N_PUSH_THREADS);
        for thread_id in 0..N_PUSH_THREADS {
            let latch = push_start_latch.clone();
            let queue = q.clone();
            push_threads.push(std::thread::spawn(move || {
                let mut my_items = Vec::<MultithreadQueueItem>::with_capacity(N_ITEMS);
                latch.on_enter();
                for n in 0..N_ITEMS {
                    let item = MultithreadQueueItem::new(n, thread_id);
                    my_items.push(item);
                    queue.push(item);
                }
                my_items
            }));
        }
        push_start_latch.await_entered();
        let mut all_pushed = Vec::with_capacity(N_PUSH_THREADS * N_ITEMS);
        for thread in push_threads {
            let items = thread.join().expect("Error waiting for thread exit");
            all_pushed.extend(items);
        }

        all_pushed.sort();

        let pop_start_latch = TestThreadLatch::<N_POP_THREADS>::new();
        let mut pop_threads = Vec::with_capacity(N_POP_THREADS);
        for _ in 0..N_POP_THREADS {
            let queue = q.clone();
            let latch = pop_start_latch.clone();
            pop_threads.push(std::thread::spawn(move || {
                let mut popped = Vec::with_capacity(N_ITEMS);
                latch.on_enter();
                while !queue.is_empty() {
                    if let Some(item) = queue.pop() {
                        popped.push(item);
                    }
                }
                popped
            }));
        }
        pop_start_latch.await_entered();

        let mut all_popped = Vec::with_capacity(N_PUSH_THREADS * N_ITEMS);
        let mut per_thread = Vec::with_capacity(N_POP_THREADS);
        for thread in pop_threads {
            let popped = thread.join().expect("Error waiting for thread exit");
            per_thread.push(popped.len());
            all_popped.extend(popped);
        }

        all_popped.sort();

        assert_eq!(all_pushed, all_popped);
        let mut sum = 0;
        for (ix, p) in per_thread.into_iter().enumerate() {
            // This may be a little iffy to test this way, since it is *possible* to just get unlucky
            // with the OS's thread-scheduler.
            assert_ne!(
                0, p,
                "Thread {ix} popped no items.  Test does not appear to be concurrent at runtime."
            );
            sum += p;
        }

        assert_eq!(N_PUSH_THREADS * N_ITEMS, sum);
    }

    #[test]
    fn test_concurrent_pushes_and_pops() {
        const N_PUSH_THREADS: usize = 5;
        const N_POP_THREADS: usize = 4;
        const TOTAL_THREADS: usize = N_PUSH_THREADS + N_POP_THREADS;
        const N_ITEMS: usize = 500_000;

        let push_threads_exited = Arc::new(AtomicUsize::default());
        let all_threads_latch = TestThreadLatch::<TOTAL_THREADS>::new();

        let mut push_threads = Vec::with_capacity(N_PUSH_THREADS);
        let mut pop_threads = Vec::with_capacity(N_POP_THREADS);

        let q = Arc::new(TreiberQueue::<MultithreadQueueItem>::new());

        for thread_id in 0..N_PUSH_THREADS {
            let latch = all_threads_latch.clone();
            let pte = push_threads_exited.clone();
            let queue = q.clone();
            push_threads.push(std::thread::spawn(move || {
                let mut my_items = Vec::<MultithreadQueueItem>::with_capacity(N_ITEMS);
                latch.on_enter();
                for n in 0..N_ITEMS {
                    let item = MultithreadQueueItem::new(n, thread_id);
                    my_items.push(item);
                    queue.push(item);
                }
                pte.fetch_add(1, SeqCst);
                my_items
            }));
        }
        for _ in 0..N_POP_THREADS {
            let latch = all_threads_latch.clone();
            let pte = push_threads_exited.clone();
            let queue = q.clone();
            pop_threads.push(std::thread::spawn(move || {
                let mut popped = Vec::with_capacity(N_ITEMS);
                let mut done_checker = DelayedDoneLatch::<N_PUSH_THREADS>::new(pte);
                latch.on_enter();
                while !done_checker.is_done() {
                    while let Some(item) = queue.pop() {
                        popped.push(item);
                    }
                    std::thread::yield_now();
                }
                popped
            }));
        }
        all_threads_latch.await_entered();

        let mut all_pushed = Vec::with_capacity(N_PUSH_THREADS * N_ITEMS);
        for thread in push_threads {
            let items = thread.join().expect("Error waiting for thread exit");
            all_pushed.extend(items);
        }

        let mut all_popped = Vec::with_capacity(N_PUSH_THREADS * N_ITEMS);
        let mut per_thread = Vec::with_capacity(N_POP_THREADS);
        for thread in pop_threads {
            let popped = thread.join().expect("Error waiting for thread exit");
            per_thread.push(popped.len());
            all_popped.extend(popped);
        }

        all_pushed.sort();
        all_popped.sort();

        assert_eq!(all_pushed, all_popped);
        let mut sum = 0;
        for (ix, p) in per_thread.into_iter().enumerate() {
            // This may be a little iffy to test this way, since it is *possible* to just get unlucky
            // with the OS's thread-scheduler.
            assert_ne!(
                0, p,
                "Thread {ix} popped no items.  Test does not appear to be concurrent at runtime."
            );
            sum += p;
        }
        assert_eq!(N_PUSH_THREADS * N_ITEMS, sum);
    }

    #[test]
    fn test_push_reports_state_correctly() {
        let q = TreiberQueue::<usize>::new();
        (0_usize..5).for_each(|ix| assert_eq!(ix == 0, q.push(ix)));
        assert!(!q.is_empty());
        let drained = q.drain();
        assert!(q.is_empty());
        assert_eq!(drained, vec![4, 3, 2, 1, 0]);
        assert!(q.push(23));
    }

    #[test]
    fn test_contents_are_dropped() {
        #[derive(Clone, Debug)]
        struct Droplet {
            index: usize,
            drop_count: Arc<AtomicUsize>,
        }

        impl Drop for Droplet {
            fn drop(&mut self) {
                self.drop_count.fetch_add(1, SeqCst);
            }
        }

        const N_ITEMS: usize = 500;

        let drops = Arc::new(AtomicUsize::new(0));
        {
            let q = TreiberQueue::<Droplet>::new();
            for index in 0..N_ITEMS {
                let first = q.push(Droplet {
                    index,
                    drop_count: drops.clone(),
                });
                assert_eq!(first, index == 0);
            }
            assert!(!q.is_empty());
            let popped = q.pop().expect("Just filled queue should pop non-None");
            assert_eq!(N_ITEMS - 1, popped.index, "Queue order is skewed");
            assert!(!q.is_empty(), "Queue should have more than one item");
        }
        assert_eq!(
            N_ITEMS,
            drops.load(SeqCst),
            "All items placed in the queue should have been dropped when the queue was dropped."
        );
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
    struct MultithreadQueueItem {
        id: u32,
        from_thread: u8,
    }

    impl MultithreadQueueItem {
        const fn new(id: usize, from_thread: usize) -> Self {
            Self {
                id: id as u32,
                from_thread: from_thread as u8,
            }
        }
    }

    /// For the concurrent pop test, we want to keep the pop threads looping a
    /// little longer than past the point at which all of the push threads report
    /// themselves to have exited, to be sure (if we switch to a less onerous guarantee
    /// than `SeqCst`) that we don't see an empty queue and exit due to cache locality,
    /// not because a value is not about to be come visible to this pop thread but isn't yet.
    struct DelayedDoneLatch<const N: usize> {
        value: Arc<AtomicUsize>,
        observed_done_at: Option<Instant>,
    }

    impl<const N: usize> DelayedDoneLatch<N> {
        fn new(value: Arc<AtomicUsize>) -> Self {
            Self {
                value,
                observed_done_at: None,
            }
        }

        fn is_done(&mut self) -> bool {
            let value_done = self.value.load(SeqCst) >= N;
            if value_done {
                let now = Instant::now();
                let when = if let Some(when) = self.observed_done_at {
                    when
                } else {
                    self.observed_done_at = Some(now);
                    now
                };
                let deadline = when + Duration::from_millis(100);
                now > deadline
            } else {
                false
            }
        }
    }

    // Good enough for an insanely loaded machine.
    const TIMEOUT: Duration = Duration::from_secs(10);

    /// A simple coordination tool using busywaits to ensure multithreaded contention tests
    /// don't run serially simply because each thread exits too fast.
    #[derive(Default)]
    struct TestThreadLatch<const N_THREADS: usize> {
        entry_count: AtomicUsize,
        released: AtomicBool,
    }

    impl<const N_THREADS: usize> TestThreadLatch<N_THREADS> {
        fn new() -> Arc<Self> {
            Arc::new(Self::default())
        }

        fn on_enter(&self) {
            self.entry_count.fetch_add(1, SeqCst);
            self.await_release();
        }

        fn release(&self) {
            self.released.store(true, SeqCst);
        }

        fn await_entered(&self) {
            // a simple busywait is more than adequate for this
            let then = Instant::now();
            while !self.entry_count.load(Acquire) == N_THREADS {
                std::thread::sleep(Duration::from_micros(30));
                if then.elapsed() > TIMEOUT {
                    panic!("Timeout exceeded");
                }
            }
            self.release();
        }

        fn await_release(&self) {
            let then = Instant::now();
            while !self.released.load(Acquire) {
                std::thread::sleep(Duration::from_micros(30));
                if then.elapsed() > TIMEOUT {
                    panic!("Timeout exceeded");
                }
            }
        }
    }

    #[derive(Debug, Eq, PartialEq, Clone)]
    struct Thing {
        id: usize,
        when: SystemTime,
        msg: String,
    }

    impl Thing {
        fn new(id: usize) -> Self {
            Thing {
                id,
                when: SystemTime::now(),
                msg: format!("Thing-{}", id),
            }
        }
    }
}
