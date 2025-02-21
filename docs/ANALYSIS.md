# Matrix Multiplication: Comparing AMD GPU vs CPU Implementation

Hey! This is my theoretical analysis comparing different ways to do matrix multiplication on AMD GPUs vs CPUs. Quick disclaimer: I wasn't able to actually run these implementations because I don't have access to a Linux machine with an AMD GPU. But I've done a deep dive into how these would work in theory and what kind of performance we might expect.

## What I'm Trying to Do Here

I've written code for four different approaches:
1. Basic CPU version (the straightforward way)
2. Simple GPU version (just getting it to work)
3. Smarter GPU version with shared memory
4. Fully optimized GPU version with all the bells and whistles

## How Each Version Works

### CPU Version (The Baseline)
Just your standard triple-nested loop - nothing fancy here. It's not the fastest, but it's super clear what's happening:
- Goes through the matrices row by row
- No special optimizations
- Easy to understand but probably pretty slow

### Basic GPU Version
This is like the "Hello World" of GPU programming:
- Each GPU thread handles one output element
- Keeps things simple but hits the memory hard
- Probably better than CPU but not by as much as you'd think

### Shared Memory Version
This is where it gets interesting:
- Uses GPU's shared memory as a sort of mini-cache
- Breaks the matrices into smaller chunks (tiles)
- Should be way faster because we're not constantly hitting global memory
- Needs threads to wait for each other sometimes (synchronization)

### Optimized Version
Pulled out all the stops here:
- Built on the shared memory version
- Unrolled some loops (4x) to reduce overhead
- Better register usage
- Tweaked the block sizes for what should be optimal performance

## What I Think Would Happen

Since I couldn't run the actual benchmarks, here's what I expect we'd see:
- CPU version would be okay for small matrices but fall off hard as they get bigger
- Basic GPU version would show some speedup but nothing crazy
- Shared memory version should be significantly better
- Optimized version would probably be the fastest, especially for larger matrices

[Note: The benchmark graphs would go here if I could run the code]

## The Big Picture

Even though I couldn't run the code, this project shows how you'd progressively optimize GPU code, specifically for AMD hardware. The cool thing about using HIP instead of CUDA is that it's basically the same concepts but with AMD's twist on things.

## What I'd Do Next

If I had access to the right hardware, I'd love to:
1. Actually run these benchmarks and see if my predictions are right
2. Play with different tile sizes to find the sweet spot
3. Maybe try some even fancier optimizations

## Setup I'd Need to Run This

Just for reference, here's what you'd need to actually run this:
- A Linux box (tried on my Mac, no dice)
- An AMD GPU
- ROCm and HIP installed
- C++ compiler that plays nice with all this

## Sources I Used

1. AMD's ROCm docs (super helpful for understanding the HIP stuff)
2. Various matrix multiplication optimization papers
3. Online GPU programming guides
4. Stack Overflow (obviously 😅)

## Important Note
Just to be super clear - this is all theoretical since I don't have access to AMD GPU hardware. The code is written and should work, but I haven't been able to verify it with actual runs. Think of it more as a "here's how you'd do it" rather than "here's how it performs." 