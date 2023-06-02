# HOP - Header Only Porting

A light-weight header-only library for GPU porting between CUDA and HIP.

> No code modifications needed. Just add a few extra compile flags to hop
> between CUDA and HIP!

Works by automatically redefining identifiers at compile time and catching
include statements in C and C++ (also Fortran with ISO C bindings).


## Install

```
export HOP_ROOT=/path/to/install/hop
git clone https://github.com/mlouhivu/hop $HOP_ROOT
```


## Usage

### Compile flags

Translate from CUDA to HIP:
```
-x hip -I$HOP_ROOT -I$HOP_ROOT/source/cuda -DHOP_TARGET_HIP
```

Translate from HIP to CUDA:
```
-x cu -I$HOP_ROOT -I$HOP_ROOT/source/hip -DHOP_TARGET_CUDA
```

#### List of compile flags

- Include path for HOP headers:<br>
  `-I$HOP_ROOT`
- Catch source code headers:<br>
  `-I$HOP_ROOT/source/cuda` or `-I$HOP_ROOT/source/hip`
- Define target GPU backend:<br>
  `-DHOP_TARGET_HIP` or `-DHOP_TARGET_CUDA`
- (optional) Override automatic filetype detection:<br>
  `-x cu` (if target is CUDA) or `-x hip` (if target is HIP)
- (optional) Define source language (e.g. if no headers are included in the
  source code):<br>
  `-DHOP_SOURCE_CUDA` or `-DHOP_SOURCE_HIP`
- (optional) Manually include HOP headers (e.g. if some header includes are
  missing in the source code):<br>
  `-include $HOP_ROOT/hop/hop_runtime.h` or similar


## Examples

### CUDA ⇒ HIP

```
export HOP_FLAGS=-I$HOP_ROOT -I$HOP_ROOT/source/cuda -DHOP_TARGET_HIP
$CC -x hip $HOP_FLAGS foo.cu -o foo
```

### HIP ⇒ CUDA

```
export HOP_FLAGS=-I$HOP_ROOT -I$HOP_ROOT/source/hip -DHOP_TARGET_CUDA
$CC -x cu $HOP_FLAGS foo.cpp -o foo
```

where `$CC` is the compiler to be used.
