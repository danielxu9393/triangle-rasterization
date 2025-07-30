# Triangle Rasterization

## Code Structure

The main rendering function is `render` in `triangle_rasterization/__init__.py`. This function combines a handful of operations (projection, alpha compositing, etc.), each of which follows an interface defined in `triangle_rasterization/interface`. Most operations have two implementations—one in PyTorch and one in Slang—which allows fast Slang operations to be tested against their easier-to-understand PyTorch counterparts. The PyTorch and Slang implementations of the operations are defined in `triangle_rasterization/torch` and `triangle_rasterization/slang` respectively.

## Workflow

Make sure you install the following software:

- **`clang-format`:** `sudo apt install clang-format`
- **The VS Code Slang Extension:** `shader-slang.slang-language-extension`
- **Ninja:** For some reason, in order to discover tests in VS Code, this has to be install via `sudo apt install ninja-build` rather than via Pip.

## Why is my function not differentiable?

- Confirm that your function and its sub-functions are marked with `[Differentiable]`.
- Confirm that any structs returned by sub-functions inherit from `IDifferentiable`.
- Note: `const` vs. non-`const` (or alternatively, `let` vs. `var`) makes no difference.
