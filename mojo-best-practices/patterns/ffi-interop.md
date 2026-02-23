---
title: FFI and C Interoperability
description: Core FFI patterns including C strings, libc functions, binary data, integer type safety, string handling, and dynamic library loading
impact: CRITICAL
category: ffi
tags: [ffi, c-interop, strings, binary, libc, dynamic-loading]
error_patterns:
  - "undefined symbol"
  - "linker error"
  - "CString"
  - "DLHandle"
  - "library not found"
  - "symbol not found"
  - "execution crashed"
  - "sockaddr"
  - "platform_map"
scenarios:
  - "Call C library from Mojo"
  - "Handle C strings safely"
  - "Fix linker error"
  - "Work with binary data"
  - "Fix integer size mismatch"
  - "Load shared library at runtime"
  - "Cross-platform socket struct layout"
  - "Fix Linux CI crash before tests run"
consolidates:
  - ffi-cstring-safety.md
  - ffi-libc-functions.md
  - ffi-binary-data-patterns.md
  - ffi-int-size-mismatch.md
  - ffi-missing-math-functions.md
  - ffi-dynamic-library-loading.md
  - ffi-string-handling.md
---

# FFI and C Interoperability

**Category:** ffi | **Impact:** CRITICAL

Core patterns for interfacing Mojo with C libraries and system APIs. Covers string handling, binary data, type safety, and dynamic library loading. For vendor libraries (BLAS, cuBLAS, MPS) and GPU integration, see [`ffi-vendor.md`](ffi-vendor.md).

---

## Core Concepts

### Type Aliases for FFI

Define standard type aliases for consistent FFI code.

**Pattern:**

```mojo
# nocompile
from ffi import external_call, OwnedDLHandle, RTLD
from memory import UnsafePointer, memcpy
from builtin.type_aliases import MutAnyOrigin

# Standard pointer types
comptime UInt8Ptr = UnsafePointer[mut=True, type=UInt8, origin=MutAnyOrigin]
comptime Float32Ptr = UnsafePointer[mut=True, type=Float32, origin=MutAnyOrigin]
comptime Float64Ptr = UnsafePointer[mut=True, type=Float64, origin=MutAnyOrigin]
comptime Int32Ptr = UnsafePointer[mut=True, type=Int32, origin=MutAnyOrigin]
comptime NonePtr = UnsafePointer[mut=True, type=NoneType, origin=MutAnyOrigin]

# Opaque pointer for FILE*
comptime FILE = UnsafePointer[mut=True, type=UInt8, origin=MutAnyOrigin]
```

### Integer Type Sizes

Mojo's `Int` is 64-bit on most platforms, but C's `int` is 32-bit. This mismatch causes silent data corruption.

**Pattern:**

```mojo
# Type size reference
# | Mojo Type | Size on Apple Silicon | C Equivalent |
# |-----------|----------------------|--------------|
# | Int       | 64-bit               | long/int64_t |
# | Int32     | 32-bit               | int/int32_t  |
# | Int64     | 64-bit               | int64_t      |
# | Int16     | 16-bit               | short        |
# | Int8      | 8-bit                | char         |
```

---

## CString Safety Patterns

### String to C Pointer — Use `unsafe_ptr()` Directly

The correct way to pass a Mojo `String` to a C function is via `s.unsafe_ptr()`.
Do **not** chain through `as_c_string_slice()` — it is a mutating method and cannot
be called on an rvalue/temporary.

```mojo
# nocompile
fn call_c_function(s: String):
    # CORRECT: direct pointer to null-terminated buffer
    external_call["c_func", NoneType](s.unsafe_ptr())

    # WRONG — mutating method on rvalue:
    # external_call["c_func", NoneType](s.as_c_string_slice().unsafe_ptr())
    # Error: invalid use of mutating method on rvalue of type 'String'
```

**Rule:** Use `s.unsafe_ptr()` for any `String → C char*` pass. The pointer is valid
for the lifetime of `s`.

### String to C String (for calling C functions)

```mojo
# nocompile
fn call_c_function(s: String):
    # Get null-terminated pointer (valid while s is alive)
    var c_str = s.unsafe_cstr_ptr()

    # Call C function
    external_call["puts", NoneType](c_str)

    # WARNING: c_str is only valid while s exists
    # Do NOT store or return c_str
```

### Correct Lifetime Management

```mojo
# nocompile
fn safe_c_call(s: String):
    # Explicitly manage lifetime
    var c_str: UnsafePointer[C_char] = s.unsafe_cstr_ptr()

    # Call is safe - s still in scope
    var result = external_call["strlen", Int](c_str)

    # s goes out of scope after this function
    # c_str becomes dangling - do not use after return
```

### C String to Mojo String (taking ownership)

```mojo
# nocompile
fn receive_c_string() -> String:
    # C function returns allocated string
    var c_str = external_call["get_message", UnsafePointer[C_char]]()

    # Option 1: Copy to String (C still owns memory)
    var s = String(StringRef(c_str))

    # Must free C memory if we own it
    external_call["free", NoneType](c_str)

    return s
```

### `StringSlice` vs `String` — String Method Return Types

Many `String` methods return `StringSlice` (a non-owning view), not an owned `String`.
This is a common FFI/API surprise in Mojo nightly.

```mojo
# nocompile
fn example(line: String):
    # These return StringSlice, NOT String:
    var a = line[:10]         # StringSlice
    var b = line.strip()      # StringSlice
    var c = line.lstrip()     # StringSlice

    # Wrap with String(...) to get owned String:
    var owned_a = String(line[:10])
    var owned_b = String(line.strip())

    # Chaining: convert at each step
    var header_key = String(String(line[:colon]).strip())  # slice → String → strip → String
```

**Rule:** Whenever you need to pass a string expression result (from slicing, `.strip()`,
etc.) to a function declared with `s: String`, wrap it in `String(...)`.

### `Span[Byte]` vs `List[UInt8]` — `as_bytes()` Return Type

`String.as_bytes()` returns `Span[Byte]` (non-owning view), not `List[UInt8]` (owned buffer).

```mojo
# nocompile
fn needs_list(body: List[UInt8]): ...

fn example(s: String):
    # WRONG — Span[Byte] cannot be passed as List[UInt8]:
    # needs_list(s.as_bytes())

    # CORRECT — copy into an owned List:
    var body = List[UInt8](s.as_bytes())
    needs_list(body^)
```

### StringSlice for Borrowed C Data

```mojo
fn process_c_buffer(ptr: UnsafePointer[UInt8], length: Int):
    # Create view without copying (ptr must remain valid)
    var slice = StringSlice[ImmutAnyOrigin](
        ptr=ptr.bitcast[Byte](),
        length=length,
    )

    # Use slice (zero-copy)
    print("Length:", len(slice))

    # slice becomes invalid when ptr is freed
```

### Buffer Allocation for C Output

```mojo
# nocompile
from memory import alloc

fn get_cwd() raises -> String:
    # Allocate buffer for C function to write into
    comptime BUFFER_SIZE = 4096
    var buffer = alloc[C_char](BUFFER_SIZE)

    # C function writes into buffer
    var result = external_call["getcwd", UnsafePointer[C_char]](
        buffer, BUFFER_SIZE
    )

    if not result:
        buffer.free()
        raise Error("getcwd failed")

    # Convert to String (copies data)
    var path = String(StringRef(buffer))

    # Free our buffer
    buffer.free()

    return path
```

### StaticString for Compile-Time C Strings

```mojo
# nocompile
fn use_static_string():
    # Compile-time string - guaranteed null-terminated, static lifetime
    comptime MSG: StaticString = "Hello, World!"

    # Safe to pass to C - lives forever
    var c_str = MSG.unsafe_cstr_ptr()
    external_call["puts", NoneType](c_str)
```

### Common CString Mistakes

```mojo
# nocompile
# BAD: Using pointer after String is freed
fn bad_lifetime() -> UnsafePointer[C_char]:
    var s = String("hello")
    return s.unsafe_cstr_ptr()  # s is freed, pointer dangles!

# BAD: Double-free
fn bad_ownership():
    var c_str = external_call["strdup", UnsafePointer[C_char]]("hello")
    var s1 = String(c_str, transfer_ownership=True)  # s1 will free
    var s2 = String(c_str, transfer_ownership=True)  # Double free!

# BAD: Assuming null termination
fn bad_assumption(ptr: UnsafePointer[UInt8], len: Int):
    # ptr might not be null-terminated!
    external_call["puts", NoneType](ptr.bitcast[C_char]())
```

---

## libc Function Usage

### Basic File I/O with libc

```mojo
# nocompile
from ffi import external_call
from memory import UnsafePointer
from builtin.type_aliases import MutAnyOrigin

# Define opaque pointer type for FILE*
comptime FILE = UnsafePointer[mut=True, type=UInt8, origin=MutAnyOrigin]

fn fopen(path: String, mode: String) -> FILE:
    """Open file using libc."""
    return external_call["fopen", FILE](path.unsafe_ptr(), mode.unsafe_ptr())

fn fclose(f: FILE) -> Int32:
    """Close file using libc."""
    return external_call["fclose", Int32](f)

fn fread(ptr: UnsafePointer[UInt8], size: Int, count: Int, f: FILE) -> Int:
    """Read from file using libc."""
    return Int(external_call["fread", Int64](ptr, size, count, f))

fn fwrite(ptr: UnsafePointer[UInt8], size: Int, count: Int, f: FILE) -> Int:
    """Write to file using libc."""
    return Int(external_call["fwrite", Int64](ptr, size, count, f))

fn fseek(f: FILE, offset: Int64, whence: Int32) -> Int32:
    """Seek in file using libc."""
    return external_call["fseek", Int32](f, offset, whence)

fn ftell(f: FILE) -> Int64:
    """Get file position using libc."""
    return external_call["ftell", Int64](f)
```

### Reading a File

```mojo
# nocompile
fn read_file_contents(path: String) raises -> List[UInt8]:
    """Read entire file into byte buffer."""
    var f = fopen(path, "rb")
    if not f:
        raise "Failed to open file: " + path

    # Get file size
    _ = fseek(f, 0, 2)  # SEEK_END
    var size = Int(ftell(f))
    _ = fseek(f, 0, 0)  # SEEK_SET

    # Read contents
    var buffer = List[UInt8](capacity=size)
    buffer.resize(size, 0)
    var bytes_read = fread(buffer.unsafe_ptr(), 1, size, f)
    _ = fclose(f)

    if bytes_read != size:
        raise "Failed to read complete file"

    return buffer
```

---

## Binary Data Patterns

### Byte-Order Functions (Little-Endian)

```mojo
# nocompile
fn read_le16(data: UInt8Ptr) -> UInt16:
    """Read little-endian 16-bit value."""
    return UInt16(data[0]) | (UInt16(data[1]) << 8)

fn read_le32(data: UInt8Ptr) -> UInt32:
    """Read little-endian 32-bit value."""
    return UInt32(data[0]) | (UInt32(data[1]) << 8) | \
           (UInt32(data[2]) << 16) | (UInt32(data[3]) << 24)

fn read_le64(data: UInt8Ptr) -> UInt64:
    """Read little-endian 64-bit value."""
    return UInt64(read_le32(data)) | (UInt64(read_le32(data + 4)) << 32)

fn write_le32(data: UInt8Ptr, val: UInt32):
    """Write little-endian 32-bit value."""
    data[0] = UInt8(val & 0xFF)
    data[1] = UInt8((val >> 8) & 0xFF)
    data[2] = UInt8((val >> 16) & 0xFF)
    data[3] = UInt8((val >> 24) & 0xFF)
```

### Byte-Order Functions (Big-Endian)

```mojo
# nocompile
fn read_be16(data: UInt8Ptr) -> UInt16:
    """Read big-endian 16-bit value."""
    return (UInt16(data[0]) << 8) | UInt16(data[1])

fn read_be32(data: UInt8Ptr) -> UInt32:
    """Read big-endian 32-bit value."""
    return (UInt32(data[0]) << 24) | (UInt32(data[1]) << 16) | \
           (UInt32(data[2]) << 8) | UInt32(data[3])

fn write_be32(data: UInt8Ptr, val: UInt32):
    """Write big-endian 32-bit value."""
    data[0] = UInt8((val >> 24) & 0xFF)
    data[1] = UInt8((val >> 16) & 0xFF)
    data[2] = UInt8((val >> 8) & 0xFF)
    data[3] = UInt8(val & 0xFF)
```

### Memory Copy (CRITICAL: Use Keyword Arguments)

```mojo
# nocompile
# WRONG: Positional arguments (deprecated, may break)
fn copy_wrong(dest: UInt8Ptr, src: UInt8Ptr, count: Int):
    memcpy(dest, src, count)  # WARNING: deprecated

# CORRECT: Keyword arguments (required in Mojo 0.26+)
fn copy_correct(dest: UInt8Ptr, src: UInt8Ptr, count: Int):
    memcpy(dest=dest, src=src, count=count)
```

### Bitcasting Between Types

```mojo
# nocompile
fn float32_to_bytes(val: Float32) -> UInt32:
    """Reinterpret float bits as integer."""
    var ptr = UnsafePointer.address_of(val)
    return ptr.bitcast[UInt32]()[]

fn bytes_to_float32(val: UInt32) -> Float32:
    """Reinterpret integer bits as float."""
    var ptr = UnsafePointer.address_of(val)
    return ptr.bitcast[Float32]()[]

fn read_float32_le(data: UInt8Ptr) -> Float32:
    """Read little-endian float32."""
    return bytes_to_float32(read_le32(data))
```

### Common Binary Formats

| Format | Byte Order | Header |
|--------|------------|--------|
| Safetensors | Little-endian | 8-byte size + JSON |
| PNG | Big-endian | 8-byte signature |
| PPM | ASCII | "P6\n" or "P3\n" |
| BMP | Little-endian | "BM" + sizes |
| GGUF | Little-endian | Magic + version |

---

## Integer Size Mismatch (CRITICAL)

### The Problem

Mojo's `Int` is 64-bit, but C's `int` is 32-bit. Direct bitcast causes data corruption.

```mojo
# nocompile
# WRONG: Direct bitcast of 64-bit Int array to 32-bit pointer
var mojo_ints: UnsafePointer[Int] = ...  # 64-bit values
var c_ints = mojo_ints.bitcast[Int32]()  # Reinterprets bytes, doesn't convert!

# Memory layout of [1, 1, 1, 1] as 64-bit:
# 0x01 0x00 0x00 0x00 0x00 0x00 0x00 0x00  (first Int = 1)
# 0x01 0x00 0x00 0x00 0x00 0x00 0x00 0x00  (second Int = 1)

# After bitcast, C sees these as 32-bit ints:
# int[0] = 1, int[1] = 0, int[2] = 1, int[3] = 0  <- WRONG!
```

### The Fix

```mojo
from memory import alloc

fn pass_int_array_to_c(mojo_ints: UnsafePointer[Int], count: Int) -> Int32Ptr:
    """Convert Mojo Int array to C-compatible Int32 array."""
    # Allocate new buffer with correct element size
    var c_ints = alloc[Int32](count)

    # Explicit element-by-element conversion
    for i in range(count):
        c_ints[i] = Int32(mojo_ints[i])

    return c_ints

# Usage:
var mask32 = pass_int_array_to_c(attention_mask, seq_len)
defer mask32.free()  # Don't forget to free!

# Now safe to pass to C/Metal FFI
var success = mps_gpu_flash_attention(output, q, k, v, mask32, ...)
```

### Debugging This Issue

Symptoms:
- GPU kernel produces wrong results but compiles without errors
- Every other element in the array appears to be zero
- Results look "randomly wrong" but are deterministically incorrect
- CPU version works fine, GPU version doesn't

```mojo
# nocompile
# Debug print to catch this issue
print("  [DEBUG MASK] attention_mask[0..3]:", attention_mask[0], attention_mask[1], attention_mask[2], attention_mask[3])
print("  [DEBUG MASK32] mask32[0..3]:", mask32[0], mask32[1], mask32[2], mask32[3])
```

---

## String Handling for C Interop

### String Indexing

```mojo
# nocompile
# WRONG: Direct string indexing
fn get_char_wrong(s: String, i: Int) -> UInt8:
    return s[i]  # ERROR: no matching method '__getitem__'

# CORRECT: Use as_bytes() for byte access
fn get_char(s: String, i: Int) -> UInt8:
    return s.as_bytes()[i]

# Iterating over string bytes
fn process_bytes(s: String):
    var bytes = s.as_bytes()
    for i in range(len(bytes)):
        var byte = bytes[i]
        # process byte...
```

### String Conversion

```mojo
# nocompile
# WRONG: Using str() for conversion
fn int_to_string_wrong(val: Int) -> String:
    return str(val)  # ERROR: 'str' not found

# CORRECT: Use String() constructor
fn int_to_string(val: Int) -> String:
    return String(val)

fn float_to_string(val: Float64) -> String:
    return String(val)

fn build_message(count: Int, name: String) -> String:
    return "Found " + String(count) + " items for " + name
```

### Writing Strings to Files

```mojo
# nocompile
fn write_string_to_file(f: FILE, s: String):
    """Write string to file using libc fwrite."""
    # String.unsafe_ptr() returns pointer to internal char data
    # bitcast to UInt8 for fwrite
    _ = fwrite(s.unsafe_ptr().bitcast[UInt8](), 1, len(s), f)

# Example: Writing PPM header
fn write_ppm_header(f: FILE, width: Int, height: Int):
    write_string_to_file(f, "P6\n")
    write_string_to_file(f, String(width) + " " + String(height) + "\n")
    write_string_to_file(f, "255\n")
```

---

## Dynamic Library Loading

### Load Library and Get Functions

```mojo
# nocompile
from ffi import OwnedDLHandle, RTLD

# Function type aliases
comptime CreateCtxFn = fn() -> NonePtr
comptime DestroyCtxFn = fn(NonePtr) -> None
comptime ProcessFn = fn(NonePtr, Int32) -> Int32

fn main() raises:
    # Load the dynamic library
    var lib = OwnedDLHandle("./libcustom.so", RTLD.NOW)  # .dylib on macOS

    # Get function pointers
    var create_ctx = lib.get_function[CreateCtxFn]("create_context")
    var destroy_ctx = lib.get_function[DestroyCtxFn]("destroy_context")
    var process = lib.get_function[ProcessFn]("process_data")

    # Create context and use
    var ctx = create_ctx()
    var result = process(ctx, 42)
    destroy_ctx(ctx)
```

### Platform-Specific Loading

```mojo
# nocompile
fn load_platform_library() raises -> OwnedDLHandle:
    """Load appropriate library for current platform."""
    @parameter
    if os_is_macos():
        return OwnedDLHandle("./libcustom.dylib", RTLD.NOW)
    elif os_is_linux():
        return OwnedDLHandle("./libcustom.so", RTLD.NOW)
    else:
        return OwnedDLHandle("./custom.dll", RTLD.NOW)
```

### Building Shared Libraries

```bash
# Linux
gcc -shared -fPIC -o libcustom.so custom.c

# macOS
clang -dynamiclib -o libcustom.dylib custom.c
```

### CRITICAL: Never Import Inside a Function Body (Linux x86_64 Crash)

`from module import X` inside a function body causes a Mojo **runtime crash**
(`error: execution crashed`) on Linux x86_64 **before any code runs**, even
if the function is never called. macOS arm64 does not crash. This is
platform-specific Mojo runtime behavior, verified in production CI.

```mojo
# nocompile
# BAD — crashes at startup on Linux x86_64:
fn find_my_lib() -> String:
    from pathlib import Path          # ← CRASH on Linux before any test runs
    if Path("build/mylib.so").exists():
        return "build/mylib.so"
    return getenv("LIB_PATH", "mylib.so")

# GOOD — all imports at the top of the file:
from os import getenv
# pathlib not needed; use env var or CONDA_PREFIX logic instead
fn find_my_lib() -> String:
    var explicit = getenv("MY_LIB", "")
    if explicit:
        return explicit
    return "build/mylib.so"
```

**Rule:** Keep all `from X import Y` at module level. Never inside functions.

### Resolving Library Path with Environment Variable (pixi/conda pattern)

When a shared library is built by an activation script (e.g., pixi's
`[activation].scripts`), export the library path as an env var. Since
activation scripts are **sourced**, `export` persists to every `pixi run …`
child process, eliminating any runtime filesystem check in Mojo.

```bash
# build.sh — sourced by pixi activation
TARGET="$SCRIPT_DIR/../../../build/libmylib.so"
# ... build steps ...
export MY_LIB="$TARGET"  # persists to all `pixi run ...` child processes

# Also export on the idempotency early-return path:
if ! _needs_rebuild; then
    export MY_LIB="$TARGET"
    return 0
fi
```

```mojo
# nocompile
# mylib.mojo — no filesystem check, no pathlib import
from os import getenv

fn find_my_lib() -> String:
    """Resolve path to libmylib.so.

    Search order:
    1. $MY_LIB      — set by activation script (dev)
    2. $CONDA_PREFIX/lib/  — installed conda package
    3. build/        — bare checkout without conda
    """
    var explicit = getenv("MY_LIB", "")
    if explicit:
        return explicit
    var prefix = getenv("CONDA_PREFIX", "")
    if prefix:
        return prefix + "/lib/libmylib.so"
    return "build/libmylib.so"
```

### dlopen Path Sensitivity for Libraries with Global State

`OwnedDLHandle` calls `dlopen`/`dlclose`. For libraries with global state
(OpenSSL, CUDA contexts, etc.), loading an **identical binary from a different
path** can produce different runtime behavior. This happens because `dlopen`
tracks identity by (device, inode), so a copy at a new path is a separate
library instance with independent global state.

```mojo
# nocompile
# PROBLEM: copying libssl_wrapper.so to $CONDA_PREFIX/lib/ and loading it from
# there vs loading from build/ causes different OpenSSL SSL_CTX behavior even
# though the bytes are identical (same md5sum). TLS tests pass from build/,
# fail from $CONDA_PREFIX/lib/.

# SOLUTION: always load from the same canonical path (use env var pattern above).
# Do NOT copy shared libs to a new path and load from there — keep a single path.
```

**Rule:** For any library with global state (`dlopen`/`dlclose` cycle), load
from a single canonical path. Use an env var set at build time rather than
copying the file to a new location at runtime.

### Cross-Platform `sockaddr_in` Layout (macOS vs Linux)

macOS (BSD) prepends a 1-byte `sin_len` field before `sin_family`; Linux
does not. Passing a macOS-layout buffer to Linux's `connect()`/`bind()`
silently connects to the wrong address. Use `CompilationTarget.is_macos()`
to branch at compile time.

```mojo
# nocompile
from sys.info import CompilationTarget

fn _fill_sockaddr_in(
    buf: UnsafePointer[UInt8], port: UInt16, ip_bytes: UnsafePointer[UInt8]
):
    """Populate a 16-byte IPv4 sockaddr_in buffer in-place."""
    @parameter
    if CompilationTarget.is_macos():
        # BSD-style: byte 0 = sin_len (struct size), byte 1 = AF_INET
        (buf + 0).init_pointee_copy(UInt8(16))  # sin_len
        (buf + 1).init_pointee_copy(UInt8(2))   # AF_INET
    else:
        # Linux: sin_family as little-endian UInt16 → bytes [2, 0]
        (buf + 0).init_pointee_copy(UInt8(2))   # family low byte
        (buf + 1).init_pointee_copy(UInt8(0))   # family high byte

    # Port big-endian (bytes 2-3), IP (bytes 4-7), padding (8-15 zeroed by caller)
    (buf + 2).init_pointee_copy(UInt8(port >> 8))
    (buf + 3).init_pointee_copy(UInt8(port & 0xFF))
    for i in range(4):
        (buf + 4 + i).init_pointee_copy((ip_bytes + i).load())
```

### `platform_map` for Compile-Time Platform Constants

Socket/libc constants differ between macOS and Linux. Use `platform_map` from
`sys.info` to select the right value at compile time — zero runtime overhead.

```mojo
# nocompile
from sys.info import platform_map

comptime _pm = platform_map[T=Int, ...]

# Socket constants that differ between platforms:
comptime AF_INET6: c_int  = c_int(_pm["AF_INET6", linux=10,     macos=30]())
comptime SOL_SOCKET: c_int = c_int(_pm["SOL_SOCKET", linux=1,   macos=0xFFFF]())
comptime SO_REUSEADDR: c_int = c_int(_pm["SO_REUSEADDR", linux=2, macos=4]())
comptime SO_REUSEPORT: c_int = c_int(_pm["SO_REUSEPORT", linux=15, macos=0x0200]())
comptime SO_KEEPALIVE: c_int = c_int(_pm["SO_KEEPALIVE", linux=9, macos=8]())
comptime SO_RCVTIMEO: c_int = c_int(_pm["SO_RCVTIMEO",  linux=20, macos=0x1006]())
comptime O_NONBLOCK: c_int  = c_int(_pm["O_NONBLOCK",   linux=2048, macos=4]())
```

---

## Missing Math Functions

Some math functions are not in Mojo's math module. Implement them using primitives.

### Power Function

```mojo
from math import exp, log, sqrt

fn pow(base: Float64, exponent: Float64) -> Float64:
    """Power function - not available in math module."""
    if base <= 0:
        if base == 0 and exponent > 0:
            return 0.0
        return 0.0  # Handle edge cases
    return exp(exponent * log(base))

fn ipow(base: Int, exponent: Int) -> Int:
    """Integer power using repeated squaring - O(log n)."""
    if exponent < 0:
        return 0
    if exponent == 0:
        return 1

    var result = 1
    var b = base
    var e = exponent

    while e > 0:
        if e & 1:
            result *= b
        b *= b
        e >>= 1

    return result
```

### Available vs Missing Functions

**Available in `from math import`:**
- `sqrt`, `exp`, `log`, `log2`, `log10`
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- `floor`, `ceil`, `abs`

**Not available (implement yourself):**
| Function | Implementation |
|----------|----------------|
| `pow(a, b)` | `exp(b * log(a))` |
| `hypot(a, b)` | `sqrt(a*a + b*b)` |
| `cbrt(x)` | `exp(log(x)/3.0)` |
| `round(x)` | `floor(x + 0.5)` |

---

## Decision Guide

| Scenario | Approach | See Also |
|----------|----------|----------|
| Need C string | Use `unsafe_cstr_ptr()`, mind lifetime | CString Safety |
| Passing int arrays to C | Convert Int to Int32 explicitly | Integer Size Mismatch |
| Loading optional library | Use `OwnedDLHandle` with RTLD.NOW | Dynamic Loading |
| Binary file format | Use byte-order functions | Binary Data Patterns |
| Matrix multiply | Use vendor BLAS (10-100x faster) | [`ffi-vendor.md`](ffi-vendor.md) |
| GPU library integration | See vendor patterns | [`ffi-vendor.md`](ffi-vendor.md) |
| Python GIL management | See vendor patterns | [`ffi-vendor.md`](ffi-vendor.md) |

---

## Quick Reference

- **CString lifetime**: `unsafe_cstr_ptr()` only valid while String lives
- **Int to C**: Always convert `Int` to `Int32` for C's `int`
- **memcpy**: Use keyword arguments: `memcpy(dest=d, src=s, count=n)`
- **Binary formats**: Use explicit byte-order functions for portability
- **Dynamic loading**: Use `OwnedDLHandle` for runtime library loading

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `undefined symbol` | Library not linked or wrong symbol name | Check library path; use `nm` to verify symbol exists; ensure correct mangling |
| `library not found` | DLHandle can't locate shared library | Use full path or set `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` |
| `CString use-after-free` | CString destroyed before C function returns | Keep CString alive until C call completes; use `with` block |
| `Int vs c_int mismatch` | Mojo Int is 64-bit, C int is 32-bit | Use `Int32` or `c_int` alias for C function arguments |
| `symbol not found` | Wrong symbol name or library | Check spelling; use `nm -gU libname` to list symbols |
| `execution crashed` (Linux, before any test output) | `from X import Y` inside a function body; Linux x86_64-specific crash | Move all imports to module level; never import inside a function body |
| Library with global state (TLS/OpenSSL) behaves differently despite identical binary | Same binary loaded via `dlopen` from two different paths creates independent global state | Load from a single canonical path; use the env-var path-resolution pattern |
| Wrong connect address on Linux (works on macOS) | `sockaddr_in` has 1-byte `sin_len` prefix on macOS/BSD, absent on Linux | Use `@parameter if CompilationTarget.is_macos()` to branch buffer layout |

---

## Version-Specific Features

### v26.1+ (Stable)

| Feature | Status | Notes |
|---------|--------|-------|
| **Stack allocation** | `stack_allocation[N, T]()` | Stable |
| **Compile-time constants** | `alias` or `comptime` | Both work in v26.1+ |
| **memcpy** | `memcpy(dest=..., src=..., count=...)` | Named params recommended |
| **external_call** | `external_call[name, ret_type](args)` | Stable |
| **OwnedDLHandle** | `OwnedDLHandle` | Stable |

**Example (v26.1+):**
```mojo
from memory import memcpy, stack_allocation

fn example():
    comptime BUFFER_SIZE = 64

    var a_ptr = stack_allocation[BUFFER_SIZE, Float32]()
    var b_ptr = stack_allocation[BUFFER_SIZE, Float32]()

    # memcpy with named args (recommended)
    memcpy(dest=b_ptr, src=a_ptr, count=BUFFER_SIZE)
```

**Notes:**
- Both `alias` and `comptime` work for compile-time constants in v26.1+
- FFI core APIs (`external_call`, `OwnedDLHandle`) are stable across versions
- `memcpy` named parameters recommended in v26.1+
- CString safety patterns unchanged between versions

---

## Related Patterns

- [`ffi-vendor.md`](ffi-vendor.md) — Vendor libraries: BLAS, cuBLAS, MPS, GPU integration
- [`python-interop.md`](python-interop.md) — Python-specific interop patterns
- [`memory-ownership.md`](memory-ownership.md) — Memory safety for FFI buffers

---

## References

- [Mojo FFI Documentation](https://docs.modular.com/mojo/std/sys/ffi/)
- [C Library Reference](https://en.cppreference.com/w/c)
