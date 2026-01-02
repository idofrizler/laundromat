# Code Style Guidelines

## Documentation Style

**DO NOT add verbose docstrings with Args/Returns sections to every function.**

### Preferred Style
- No docstrings that describe what the function does
- Use inline comments for non-trivial logic or important variables

### When Comments ARE Appropriate
- Public API endpoints
- Complex algorithms that need explanation
- Functions with many parameters that aren't obvious from names/types

### Key Principle
Type hints + good function/parameter names should make Args/Returns sections unnecessary in most cases.