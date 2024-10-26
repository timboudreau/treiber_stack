/*
 * The MIT License
 *
 * Copyright 2023 Tim Boudreau.
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
use arc_swap::ArcSwapOption;
use std::{
    fmt::{Debug, Display, Write},
    sync::Arc,
};

// Convenience type
type CellInner<T> = Option<Arc<TreiberCell<T>>>;

/// A thread-safe, lockless, single-ended linked list using atomics to update the head node.
/// Since this structure can be modified concurrently at all times, all operations that calculate
/// size or retrieve contents are retrieving a snapshot-in-time of the state of the stack.
///
/// Iteration order is last-in, first-out.
///
/// As with any concurrent data structure, all operations are performed against a snapshot
/// of the structure as it existed when the operation was performed.
///
/// The key to atomic data structures is that there must be **exactly one** atomic mutation per
/// operation and exactly one place to do it - more and races become possible.  So the *only*
/// mutable element to this structure is the pointer to the head of the list.
///
/// Deletion happens only by popping elements - while it is theoretically possible to delete
/// arbitrary items (by re-stitching an entirely new linked list sans the element
/// to be deleted, *repeatedly* in the case that the head has changed, but this is subject to
/// the [ABA problem](https://en.wikipedia.org/wiki/ABA_problem) and at best, difficult to
/// prove the correctness of).
///
/// Uses the [`arc_swap` crate under the hood](https://docs.rs/arc-swap/latest/arc_swap/) - which
/// means that returned elements are in an `Arc` (a necessity of the assign-and-test nature of
/// atomics is that for non-`Copy` types, it must be possible to pass ownership of the element
/// being added more than once in the case of contention).  If you need to mutate the contents, you likely
/// need some indirection in the element type that permits you to pull a mutable instance out
/// of them, such as a `Mutex` or other smart pointer that allows for interior mutability and is
/// thread-safe.
pub struct TreiberStack<T: Send + Sync> {
    head: ArcSwapOption<TreiberCell<T>>,
}

struct TreiberCell<T: Send + Sync> {
    value: Arc<T>,
    next: CellInner<T>,
}

/// Creates a new empty Treiber stack.
impl<T: Send + Sync> Default for TreiberStack<T> {
    fn default() -> Self {
        Self {
            head: ArcSwapOption::empty(),
        }
    }
}

impl<T: Send + Sync> Into<Vec<Arc<T>>> for TreiberStack<T> {
    fn into(self) -> Vec<Arc<T>> {
        self.drain()
    }
}

impl<T: Send + Sync, I: IntoIterator<Item = J>, J: Into<T>> From<I> for TreiberStack<T> {
    fn from(value: I) -> Self {
        let result = Self::default();
        for node in value.into_iter() {
            result.push(node)
        }
        result
    }
}

impl<T: Send + Sync> TreiberStack<T> {
    /// Create a new instance with the passed object as the head.
    ///
    /// Example:
    /// ```
    /// let stack = treiber_stack::TreiberStack::initialized_with(42_usize);
    /// assert!(!stack.is_empty());
    /// assert_eq!(42_usize, *stack.peek().unwrap());
    /// ```
    pub fn initialized_with(item: T) -> Self {
        Self {
            head: ArcSwapOption::new(Some(Arc::new(TreiberCell {
                value: Arc::new(item),
                next: None,
            }))),
        }
    }

    /// Push an item onto the stack, which will become the head.
    /// Example:
    ///
    /// ```
    /// let stack = treiber_stack::TreiberStack::default();
    /// assert!(stack.is_empty()); // stack is initially empty
    /// stack.push(23_usize); // push one element
    ///
    /// assert!(!stack.is_empty()); // the stack is not empty
    ///
    /// assert_eq!(23_usize, *stack.pop().unwrap()); // pop the element back off
    /// assert!(stack.is_empty()); // the stack is back to empty
    /// ```
    pub fn push<I: Into<T>>(&self, val: I) {
        let a = Arc::new(val.into());
        self.push_arc(a);
    }

    /// Push an extant `Arc` onto the stack, which will become the head.
    /// This method is useful for cases where values are swapped between
    /// `TreiberStack`s or examined and replaced, making cloning the value
    /// out of the `Arc` just to have it put in a new `Arc` when it is pushed
    /// adds overhead.
    pub fn push_arc(&self, a: Arc<T>) {
        self.head.rcu(|old| {
            if let Some(curr_head) = old {
                let new_head = prepend(curr_head.clone(), a.clone());
                Some(Arc::new(new_head))
            } else {
                Some(Arc::new(TreiberCell {
                    value: a.clone(),
                    next: None,
                }))
            }
        });
    }

    /// Pop the head item from this stack.
    ///
    /// Example:
    /// ```
    /// let stack = treiber_stack::TreiberStack::from(vec![3_usize, 2, 1]);
    /// assert_eq!(3, stack.len());
    /// assert_eq!(1_usize, *stack.pop().unwrap());
    /// assert_eq!(2_usize, *stack.pop().unwrap());
    /// assert_eq!(3_usize, *stack.pop().unwrap());
    /// assert_eq!(None, stack.pop());
    /// assert!(stack.is_empty());
    /// ```
    pub fn pop(&self) -> Option<Arc<T>> {
        let popped = self.head.rcu(|old| match old {
            Some(curr_head) => {
                let mm = curr_head.next.clone();
                if let Some(old_next) = mm {
                    Some(old_next)
                } else {
                    None
                }
            }
            None => None,
        });
        if let Some(v) = popped {
            let result = Some(v.value.clone());
            result
        } else {
            None
        }
    }

    /// Pop a value, if any, taking it out of the Arc it is stored in internally.
    /// Only available when `T: Copy`.
    /// Example:
    /// ```
    /// let stack : treiber_stack::TreiberStack<usize> = treiber_stack::TreiberStack::from(vec![3_usize, 2, 1]);
    /// assert_eq!(Some(1_usize), stack.pop_raw());
    /// assert_eq!(Some(2_usize), stack.pop_raw());
    /// assert_eq!(Some(3_usize), stack.pop_raw());
    /// assert!(stack.pop_raw().is_none());
    /// assert!(stack.is_empty());
    /// ```
    pub fn pop_raw(&self) -> Option<T>
    where
        T: Copy,
    {
        if let Some(result) = self.pop() {
            Some(*result)
        } else {
            None
        }
    }

    /// Drain *all* items from this Treiber stack, repeatedly calling the passed `FnMut` with
    /// each item, returning the number of elements passed to `f`.  The head of the stack
    /// is replaced by an empty cell at the start of this method, so any concurrent write
    /// operations will not be reflected in the set of elements passed to `f`, and there
    /// should be no expectation that the stack is actually empty under concurrency at
    /// the end of this call.  Prefer this method to `drain_into()` where the stack
    /// should be emptied, as it cannot encounter contention after the initial head-swap.
    ///
    /// Example:
    ///
    /// ```
    /// let stack: treiber_stack::TreiberStack<usize> = treiber_stack::TreiberStack::from(vec![6_usize, 5, 4, 3, 2, 1]);
    /// let mut v = Vec::with_capacity(6);
    /// stack.drain_all_into(|item| {
    ///     v.push(*item);
    /// });
    /// assert_eq!(vec![1, 2, 3, 4, 5, 6], v);
    /// ```
    pub fn drain_all_into<F: FnMut(Arc<T>)>(&self, mut f: F) -> usize {
        let mut head = self.head.swap(None);
        let mut processed = 0_usize;
        while let Some(curr) = head {
            f(curr.value.clone());
            processed += 1;
            head = curr.next.clone()
        }
        processed
    }

    /// Drain *all* items from this Treiber stack, where items are `Copy` and can be removed
    /// from an `Arc`, repeatedly calling the passed `FnMut` with
    /// each item, returning the number of elements passed to `f`.  The head of the stack
    /// is replaced by an empty cell at the start of this method, so any concurrent write
    /// operations will not be reflected in the set of elements passed to `f`, and there
    /// should be no expectation that the stack is actually empty under concurrency at
    /// the end of this call.
    ///
    /// Example:
    ///
    /// ```
    /// let stack: treiber_stack::TreiberStack<usize>  = treiber_stack::TreiberStack::from(vec![6_usize, 5, 4, 3, 2, 1]);
    /// let mut v = Vec::with_capacity(6);
    /// stack.drain_all_into(|item| {
    ///     v.push(*item);
    /// });
    /// assert_eq!(vec![1, 2, 3, 4, 5, 6], v);
    /// ```
    pub fn drain_all_copy<F: FnMut(T)>(&self, mut f: F) -> usize
    where
        T: Copy,
    {
        let mut head = self.head.swap(None);
        let mut processed = 0_usize;
        while let Some(curr) = head {
            let val = *curr.value;
            f(val);
            processed += 1;
            head = curr.next.clone()
        }
        processed
    }

    /// Drain items from this Treiber stack, repeatedly calling the passed `FnMut` with
    /// each item until it returns false, returning the number of elements passed to
    /// `f`.
    ///
    /// Example:
    ///
    /// ```
    /// let stack = treiber_stack::TreiberStack::from(vec![6_usize, 5, 4, 3, 2, 1]);
    /// let mut v = Vec::with_capacity(6);
    /// stack.drain_into(|item| {
    ///     v.push(*item);
    ///     *item < 3
    /// });
    /// assert_eq!(vec![1, 2, 3], v);
    /// assert_eq!(3, stack.len());
    /// assert_eq!(vec![4_usize, 5, 6], stack.drain_transforming(|item| *item));
    /// ```
    pub fn drain_into<F: FnMut(Arc<T>) -> bool>(&self, mut f: F) -> usize {
        let mut processed = 0_usize;
        loop {
            if let Some(item) = self.pop() {
                processed += 1;
                if !f(item) {
                    break;
                }
            } else {
                break;
            }
        }
        processed
    }

    /// Determine if the stack currently contains no elements
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.head.load_full().is_none()
    }

    /// Get the number of elements on the stack at the time this method was called
    pub fn len(&self) -> usize {
        if let Some(head) = self.head.load_full() {
            head.len()
        } else {
            0
        }
    }

    /// Discard the contents of this stack
    ///
    /// Example:
    /// ```
    /// let stack : treiber_stack::TreiberStack<usize> = treiber_stack::TreiberStack::from(vec![1_usize, 2, 3]);
    /// assert_eq!(3, stack.len());
    /// stack.clear();
    /// assert_eq!(0, stack.len());
    /// ```
    pub fn clear(&self) {
        self.head.store(None);
    }

    /// Determine if any element contained in this stack matches the passed
    /// predicate.
    /// Example:
    /// ```
    /// let stack : treiber_stack::TreiberStack<usize> = treiber_stack::TreiberStack::from(vec![1_usize, 2, 3]);
    /// assert!(stack.contains(|item| *item == 2));
    /// assert!(!stack.contains(|item| *item == 23));
    /// ```
    pub fn contains<F: FnMut(&T) -> bool>(&self, mut predicate: F) -> bool {
        if let Some(head) = self.head.load_full().as_ref() {
            predicate(&head.value) || {
                let mut maybe_next = &head.next;
                while let Some(next) = maybe_next {
                    if predicate(&next.value) {
                        return true;
                    }
                    maybe_next = &next.next
                }
                false
            }
        } else {
            false
        }
    }

    /// Empty this stack, returning a `Vec` of its contents.  Note that the head is
    /// detached, emptying the stack *immediately*, prior to collecting the contents,
    /// so additions to the stack while iteration is in progress will not be detected.
    ///
    /// In the case that you want to be *sure* all items have been processed (say, tasks
    /// to do during process shutdown), use a while loop testing emptiness for some
    /// period of time after initiating shutdown and ensuring no further items can be
    /// added.
    /// Example:
    /// ```
    /// let stack = treiber_stack::TreiberStack::from(vec![3_usize, 2, 1]);
    /// let drained = stack.drain();
    /// assert_eq!(vec![std::sync::Arc::new(1_usize),
    ///     std::sync::Arc::new(2_usize),
    ///     std::sync::Arc::new(3_usize)], drained);
    /// ```
    pub fn drain(&self) -> Vec<Arc<T>> {
        if let Some(head) = self.head.swap(None) {
            let mut result = Vec::new();
            result.push(head.value.clone());
            let mut maybe_next = &head.next;
            while let Some(next) = maybe_next {
                result.push(next.value.clone());
                maybe_next = &next.next
            }
            result
        } else {
            vec![]
        }
    }

    /// Empty this stack, returning a `Vec` of its contents, replacing the head with the
    /// passed value in a single atomic swap.
    /// Example:
    /// ```
    /// let stack = treiber_stack::TreiberStack::from(vec![3_usize, 2, 1]);
    /// let drained = stack.drain_replace(52);
    /// assert_eq!(vec![std::sync::Arc::new(1_usize),
    ///     std::sync::Arc::new(2_usize),
    ///     std::sync::Arc::new(3_usize)], drained);    
    /// assert_eq!(1, stack.len());
    /// assert_eq!(52, *stack.pop().unwrap());
    /// ```
    pub fn drain_replace(&self, new_head: T) -> Vec<Arc<T>> {
        if let Some(head) = self.head.swap(Some(Arc::new(TreiberCell {
            value: Arc::new(new_head),
            next: None,
        }))) {
            let mut result = Vec::new();
            result.push(head.value.clone());
            let mut maybe_next = &head.next;
            while let Some(next) = maybe_next {
                result.push(next.value.clone());
                maybe_next = &next.next
            }
            result
        } else {
            vec![]
        }
    }

    /// Empty this stack, using the passed function to transform the values encountered, and
    /// returning a Vec of the result.  Note that the resulting `Vec` will be in reverse order
    /// that elements were added, in LIFO order.
    /// ```
    /// let stack : treiber_stack::TreiberStack<usize>
    ///     = treiber_stack::TreiberStack::from(vec![3_usize, 2, 1]);
    /// let drained = stack.drain_transforming(|item| *item * 10);
    /// assert_eq!(vec![10_usize, 20, 30], drained);
    /// assert!(stack.is_empty());
    /// ```
    pub fn drain_transforming<R, F: FnMut(Arc<T>) -> R>(&self, mut transform: F) -> Vec<R> {
        if let Some(head) = self.head.swap(None) {
            let mut result = Vec::new();
            result.push(transform(head.value.clone()));
            let mut nxt = &head.next;
            while let Some(next) = nxt {
                result.push(transform(next.value.clone()));
                nxt = &next.next
            }
            result
        } else {
            vec![]
        }
    }

    /// Take a snapshot of the contents *without altering the stack or removing entries*.
    /// Example:
    /// ```
    /// let stack = treiber_stack::TreiberStack::from(vec![3_usize, 2, 1]);
    /// let snapshot = stack.snapshot();
    /// assert_eq!(vec![std::sync::Arc::new(1_usize),
    ///     std::sync::Arc::new(2_usize),
    ///     std::sync::Arc::new(3_usize)], snapshot);    
    /// assert_eq!(3, stack.len());
    /// assert_eq!(1_usize, *stack.pop().unwrap());
    /// ```
    pub fn snapshot(&self) -> Vec<Arc<T>> {
        if let Some(head) = self.head.load_full().as_ref() {
            let mut result = Vec::new();
            head.copy_into(&mut result);
            result
        } else {
            vec![]
        }
    }

    /// Retrieve a copy of the head element of the stack without altering the stack.
    /// Example:
    /// ```
    /// let stack = treiber_stack::TreiberStack::from(vec![2_usize, 1]);
    /// assert_eq!(1_usize, *stack.peek().unwrap()); // should be the last added element
    /// assert_eq!(1_usize, *stack.pop().unwrap()); // head should be unaltered
    ///
    /// assert_eq!(2_usize, *stack.peek().unwrap()); // now 2 is the head
    /// assert_eq!(2_usize, *stack.pop().unwrap()); // pop that
    /// assert_eq!(None, stack.peek()); // nothing to peek at
    /// ```
    pub fn peek(&self) -> Option<Arc<T>> {
        if let Some(head) = self.head.load_full() {
            Some(head.value.clone())
        } else {
            None
        }
    }

    /// Create an iterator over this stack.  The snapshot the iterator will use is fixed at
    /// the time of creation.
    ///
    /// Example:
    /// ```
    /// let stack = treiber_stack::TreiberStack::from(vec![2_usize, 1]);
    /// let mut copied_out = Vec::with_capacity(2);
    /// // Pushing new items will *not* cause them to be visible in this iterator
    /// for item in stack.iter() {
    ///     stack.push(*item * 10);
    ///     copied_out.push(*item);
    /// }
    /// // But they will now be present in the stack
    /// assert_eq!(vec![20_usize, 10_usize, 1_usize, 2_usize], stack.drain_transforming(|item| *item));
    /// ```    
    pub fn iter(&self) -> TreiberStackIterator<T> {
        TreiberStackIterator {
            curr: self.head.load().clone(),
        }
    }

    /// Swap the head nodes between two `TreiberStack`s of the same type.
    ///
    /// Example:
    /// ```
    /// let a : treiber_stack::TreiberStack<usize> = treiber_stack::TreiberStack::from(vec![3_usize, 2, 1]);
    /// let b : treiber_stack::TreiberStack<usize> = treiber_stack::TreiberStack::from(vec![6_usize, 5, 4]);
    /// a.exchange_contents(&b);
    /// assert_eq!(4, a.pop_raw().unwrap());
    /// assert_eq!(5, a.pop_raw().unwrap());
    /// assert_eq!(6, a.pop_raw().unwrap());
    /// assert!(a.is_empty());
    /// assert_eq!(1, b.pop_raw().unwrap());
    /// assert_eq!(2, b.pop_raw().unwrap());
    /// assert_eq!(3, b.pop_raw().unwrap());
    /// assert!(b.is_empty());
    /// ```
    pub fn exchange_contents(&self, other: &Self) {
        let _ = self.head.rcu(|my_head| other.head.rcu(|_| my_head.clone()));
    }
}

impl<T: Send + Sync> TreiberCell<T> {
    fn len(&self) -> usize {
        let mut result = 1;
        // Anything like this could be simpler recursively, but risks blowing the stack.
        let mut nxt = &self.next;
        while let Some(next) = nxt {
            result += 1;
            nxt = &next.next;
        }
        result
    }

    fn copy_into(&self, into: &mut Vec<Arc<T>>) {
        into.push(self.value.clone());
        let mut nxt = &self.next;
        while let Some(next) = nxt {
            into.push(next.value.clone());
            nxt = &next.next;
        }
    }
}

/// Iterator for a Treiber stack.
pub struct TreiberStackIterator<T: Send + Sync> {
    curr: Option<Arc<TreiberCell<T>>>,
}

impl<'l, T: Send + Sync + 'l> Iterator for TreiberStackIterator<T> {
    type Item = Arc<T>;

    fn next(&mut self) -> Option<Self::Item> {
        // There is probably a cleaner way of doing this...
        let mut old: Option<Arc<TreiberCell<T>>> = None;
        std::mem::swap(&mut old, &mut self.curr);
        if let Some(node) = old {
            self.curr = node.next.clone();
            Some(node.value.clone())
        } else {
            None
        }
    }
}

fn prepend<T: Send + Sync>(old: Arc<TreiberCell<T>>, t: Arc<T>) -> TreiberCell<T> {
    let op: CellInner<T> = Some(old);
    TreiberCell { value: t, next: op }
}

impl<T: Send + Sync> TreiberCell<T>
where
    T: Display,
{
    fn stringify(&self, into: &mut String) {
        into.push_str(self.value.to_string().as_str());
        let mut nxt = &self.next;
        while let Some(next) = nxt {
            into.push(',');
            into.push_str(next.value.to_string().as_str());
            nxt = &next.next;
        }
    }
}

impl<T: Send + Sync> TreiberCell<T>
where
    T: Debug,
{
    fn debugify(&self, into: &mut String) {
        into.push_str(format!("{:?}", self.value).as_str());
        let mut nxt = &self.next;
        while let Some(next) = nxt {
            into.push(',');
            into.push_str(format!("{:?}", next.value).as_str());
            nxt = &next.next;
        }
    }
}

impl<T: Send + Sync> Display for TreiberCell<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut into = String::new();
        self.stringify(&mut into);
        f.write_str(into.as_str())
    }
}

impl<T: Send + Sync + Debug> Debug for TreiberCell<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut into = String::new();
        self.debugify(&mut into);
        f.write_str(into.as_str())
    }
}

impl<T: Send + Sync> Display for TreiberStack<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        if let Some(head) = self.head.load().as_ref() {
            let mut text = String::new();
            head.stringify(&mut text);
            f.write_str(&text)?;
        }
        f.write_char(')')
    }
}

impl<T: Send + Sync> Debug for TreiberStack<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        if let Some(head) = self.head.load().as_ref() {
            let mut text = String::new();
            head.debugify(&mut text);
            f.write_str(&text)?;
        }
        f.write_char(')')
    }
}

#[cfg(test)]
#[allow(unused_imports, dead_code, clippy::vec_init_then_push)]
mod treiber_stack_tests {
    use std::{
        fmt::Display,
        ops::Range,
        sync::{atomic::AtomicUsize, Arc},
        thread::{self, JoinHandle},
    };

    use super::TreiberStack;

    #[test]
    fn test_simple() {
        let ts: TreiberStack<&str> = TreiberStack::default();
        assert!(ts.is_empty());
        assert!(ts.peek().is_none());

        ts.push("one");
        assert!(!ts.is_empty());
        assert_eq!(1_usize, ts.len());
        assert!(ts.peek().is_some());
        assert_eq!(Some(Arc::new("one")), ts.peek());
        assert!(
            ts.contains(|o| "one".eq(*o)),
            "Not present or unequal: 'one'"
        );
        assert_eq!(1_usize, ts.len());

        ts.push("two");
        assert!(!ts.is_empty());
        assert_eq!(2_usize, ts.len());

        ts.push("three");
        assert!(!ts.is_empty());
        assert_eq!(3_usize, ts.len());

        ts.push("four");
        assert!(!ts.is_empty());
        assert_eq!(4_usize, ts.len());
        assert!(ts.peek().is_some());
        assert_eq!(Some(Arc::new("four")), ts.peek());

        let text = ts.to_string();
        assert_eq!("(four,three,two,one)", &text);

        let dbg_text = format!("{:?}", ts);
        assert_eq!("(\"four\",\"three\",\"two\",\"one\")", &dbg_text);

        let a = ts.pop();
        assert!(a.is_some());
        assert_eq!(&"four", a.as_ref().unwrap().as_ref());

        let a = ts.pop();
        assert!(a.is_some());
        assert_eq!(&"three", a.as_ref().unwrap().as_ref());

        let b = ts.pop();
        assert_eq!(&"two", b.as_ref().unwrap().as_ref());

        let c = ts.pop();
        assert_eq!(&"one", c.as_ref().unwrap().as_ref());

        assert_eq!(None, ts.pop());

        ts.clear();

        ts.push("five");
        assert_eq!(1, ts.len());
        assert!(!ts.is_empty());

        ts.clear();
        assert_eq!(0, ts.len());
        assert!(ts.is_empty());
    }

    #[test]
    fn test_from_and_into() {
        let v: Vec<usize> = vec![6, 5, 4, 3, 2, 1];
        let stack: TreiberStack<usize> = TreiberStack::from(v);
        assert!(!stack.is_empty());
        assert_eq!(6, stack.len());
        let v: Vec<Arc<usize>> = stack.into();
        assert_eq!(6, v.len());
        println!("{:?}", v);
        let mut vv: Vec<usize> = Vec::with_capacity(v.len());
        for item in v.into_iter() {
            vv.push(*item);
        }
        assert_eq!(vec![1, 2, 3, 4, 5, 6], vv);
    }

    #[test]
    fn test_pop_fn() {
        let stack: TreiberStack<usize> = TreiberStack::from(vec![6_usize, 5, 4, 3, 2, 1]);
        let mut v = Vec::with_capacity(6);
        stack.drain_into(|item| {
            v.push(*item);
            true
        });
        assert_eq!(6, v.len());
        assert_eq!(vec![1, 2, 3, 4, 5, 6], v);
    }

    #[test]
    fn test_pop_fn_filter() {
        let stack: TreiberStack<usize> = TreiberStack::from(vec![6_usize, 5, 4, 3, 2, 1]);
        let mut v = Vec::with_capacity(6);
        stack.drain_into(|item| {
            v.push(*item);
            *item < 3
        });
        assert_eq!(vec![1, 2, 3], v);
        assert_eq!(3, stack.len());
        assert_eq!(vec![4_usize, 5, 6], stack.drain_transforming(|item| *item));
    }

    #[test]
    fn test_threaded() {
        const THREADS: usize = 8;
        const MAX: usize = 1000;
        let ts: TreiberStack<Thing> = TreiberStack::default();
        let counter = AtomicUsize::new(0);
        let thread_id = AtomicUsize::new(0);
        thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    let id = thread_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let mut count: usize = 0;
                    loop {
                        let next = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        if next > MAX {
                            break;
                        }
                        // encourage some mis-ordering, and that
                        // this does not run so fast that one thread does
                        // all the work and others are starved, or we won't
                        // be testing what we think we are.
                        thread::yield_now();
                        ts.push(Thing { value: next });
                        count += 1;
                    }
                    // println!("Exit {} with {}", id, count);
                    assert!(count > 0, "No items added by thread {}", id);
                });
            }
        });

        let mut from_iter = Vec::with_capacity(ts.len());
        for item in ts.iter() {
            from_iter.push(item.value);
        }

        // println!("{}", ts);
        let copy = ts.snapshot();
        let mut from_copy = Vec::with_capacity(copy.len());
        for t in copy {
            from_copy.push(t.value);
        }
        from_copy.sort();
        let mut expected = Vec::with_capacity(MAX + 1);
        for i in 0_usize..(MAX + 1) {
            expected.push(i);
        }
        let mut got = ts.drain_transforming(|t| t.value);
        got.sort();
        from_iter.sort();
        assert_eq!(expected, got, "Contents do not match");
        assert_eq!(expected, from_copy, "Contents from copy do not match");
        assert_eq!(expected, from_iter, "Contents from iterator do not match");
        assert!(ts.is_empty(), "Should be empty");
    }

    #[derive(Debug)]
    // A type to use just to ensure tests aren't
    // fooled by any of the characteristics of Copy primitives
    struct Thing {
        value: usize,
    }

    impl Display for Thing {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.value.to_string().as_str())
        }
    }

    #[test]
    fn test_batches() {
        const THREADS: usize = 12;
        const ITEMS: usize = 15000;
        let ts: Arc<TreiberStack<usize>> = Default::default();

        fn run_range(rng: Range<usize>, stack: Arc<TreiberStack<usize>>) -> JoinHandle<()> {
            std::thread::spawn(move || {
                for i in rng {
                    stack.push(i);
                }
            })
        }

        fn ranges() -> Vec<Range<usize>> {
            let mut result = Vec::with_capacity(THREADS);
            for i in 0..THREADS {
                let start = i * ITEMS;
                let end = start + ITEMS;
                result.push(start..end);
            }
            result
        }

        let mut handles = Vec::new();
        for range in ranges() {
            handles.push(run_range(range, ts.clone()));
        }
        for h in handles {
            h.join();
        }
        let mut all = ts.drain();

        /*
        // For debugging
        for i in (all.as_slice()[0..600]).iter() {
            print!("{}, ", i);
        }
        */

        all.sort();
        let mut prev: Option<usize> = None;
        for item in all.iter() {
            let item = **item;
            if let Some(p) = prev {
                if p != item - 1 {
                    println!("Discontinuity: {} - {}", p, item);
                }
            }
            prev = Some(item);
        }
        assert_eq!(THREADS * ITEMS, all.len(), "Size mismatch");
    }
}
