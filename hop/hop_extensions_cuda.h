#if defined(_HOP_EXTENSIONS_COMPLEX)
/*
 * Operator overloading for complex data types
 *   operator: +
 *   data types: cuDoubleComplex, cuFloatComplex, <T> for scalars
 */
__host__ __device__ __inline__
cuDoubleComplex operator+(cuDoubleComplex &a, cuDoubleComplex &b)
{
    return cuCadd(a, b);
}
__host__ __device__ __inline__
cuFloatComplex operator+(cuFloatComplex &a, cuFloatComplex &b)
{
    return cuCaddf(a, b);
}
__host__ __device__ __inline__
cuDoubleComplex operator+(cuFloatComplex &a, cuDoubleComplex &b)
{
    return cuCadd(cuComplexFloatToDouble(a), b);
}
__host__ __device__ __inline__
cuDoubleComplex operator+(cuDoubleComplex &a, cuFloatComplex &b)
{
    return cuCadd(a, cuComplexFloatToDouble(b));
}

template <typename T>
__host__ __device__ __inline__
cuDoubleComplex operator+(cuDoubleComplex &a, T &b)
{
    return cuCadd(a, make_cuDoubleComplex(b, 0));
}
template <typename T>
__host__ __device__ __inline__
cuDoubleComplex operator+(T &a, cuDoubleComplex &b)
{
    return cuCadd(make_cuDoubleComplex(a, 0), b);
}
template <typename T>
__host__ __device__ __inline__
cuFloatComplex operator+(cuFloatComplex &a, T &b)
{
    return cuCaddf(a, make_cuFloatComplex(b));
}
template <typename T>
__host__ __device__ __inline__
cuFloatComplex operator+(T &a, cuFloatComplex &b)
{
    return cuCaddf(make_cuFloatComplex(a), b);
}


/*
 * Operator overloading for complex data types
 *   operator: -
 *   data types: cuDoubleComplex, cuFloatComplex, <T> for scalars
 */
__host__ __device__ __inline__
cuDoubleComplex operator-(cuDoubleComplex &a, cuDoubleComplex &b)
{
    return cuCsub(a, b);
}
__host__ __device__ __inline__
cuFloatComplex operator-(cuFloatComplex &a, cuFloatComplex &b)
{
    return cuCsubf(a, b);
}
__host__ __device__ __inline__
cuDoubleComplex operator-(cuFloatComplex &a, cuDoubleComplex &b)
{
    return cuCsub(cuComplexFloatToDouble(a), b);
}
__host__ __device__ __inline__
cuDoubleComplex operator-(cuDoubleComplex &a, cuFloatComplex &b)
{
    return cuCsub(a, cuComplexFloatToDouble(b));
}

template <typename T>
__host__ __device__ __inline__
cuDoubleComplex operator-(cuDoubleComplex &a, T &b)
{
    return cuCsub(a, make_cuDoubleComplex(b, 0));
}
template <typename T>
__host__ __device__ __inline__
cuDoubleComplex operator-(T &a, cuDoubleComplex &b)
{
    return cuCsub(make_cuDoubleComplex(a, 0), b);
}
template <typename T>
__host__ __device__ __inline__
cuFloatComplex operator-(cuFloatComplex &a, T &b)
{
    return cuCsubf(a, make_cuFloatComplex(b, 0));
}
template <typename T>
__host__ __device__ __inline__
cuFloatComplex operator-(T &a, cuFloatComplex &b)
{
    return cuCsubf(make_cuFloatComplex(a, 0), b);
}


/*
 * Operator overloading for complex data types
 *   operator: *
 *   data types: cuDoubleComplex, cuFloatComplex, <T> for scalars
 */
__host__ __device__ __inline__
cuDoubleComplex operator*(cuDoubleComplex &a, cuDoubleComplex &b)
{
    return cuCmul(a, b);
}
__host__ __device__ __inline__
cuFloatComplex operator*(cuFloatComplex &a, cuFloatComplex &b)
{
    return cuCmulf(a, b);
}
__host__ __device__ __inline__
cuDoubleComplex operator*(cuFloatComplex &a, cuDoubleComplex &b)
{
    return cuCmul(cuComplexFloatToDouble(a), b);
}
__host__ __device__ __inline__
cuDoubleComplex operator*(cuDoubleComplex &a, cuFloatComplex &b)
{
    return cuCmul(a, cuComplexFloatToDouble(b));
}

template <typename T>
__host__ __device__ __inline__
cuDoubleComplex operator*(cuDoubleComplex &a, T &b)
{
    return cuCmul(a, make_cuDoubleComplex(b, 0));
}
template <typename T>
__host__ __device__ __inline__
cuDoubleComplex operator*(T &a, cuDoubleComplex &b)
{
    return cuCmul(make_cuDoubleComplex(a, 0), b);
}
template <typename T>
__host__ __device__ __inline__
cuFloatComplex operator*(cuFloatComplex &a, T &b)
{
    return cuCmul(a, make_cuFloatComplex(b, 0));
}
template <typename T>
__host__ __device__ __inline__
cuFloatComplex operator*(T &a, cuFloatComplex &b)
{
    return cuCmul(make_cuFloatComplex(a, 0), b);
}


/*
 * Operator overloading for complex data types
 *   operator: +=
 *   data types: cuDoubleComplex, cuFloatComplex, <T> for scalars
 */
__host__ __device__ __inline__
void operator+=(cuDoubleComplex &a, cuDoubleComplex &b)
{
    a = cuCadd(a, b);
}
__host__ __device__ __inline__
void operator+=(cuFloatComplex &a, cuFloatComplex &b)
{
    a = cuCaddf(a, b);
}
__host__ __device__ __inline__
void operator+=(cuFloatComplex &a, cuDoubleComplex &b)
{
    a = cuCaddf(a, cuComplexDoubleToFloat(b));
}
__host__ __device__ __inline__
void operator+=(cuDoubleComplex &a, cuFloatComplex &b)
{
    a = cuCadd(a, cuComplexFloatToDouble(b));
}

template <typename T>
__host__ __device__ __inline__
void operator+=(cuDoubleComplex &a, T &b)
{
    a = cuCadd(a, make_cuDoubleComplex(b));
}
template <typename T>
__host__ __device__ __inline__
void operator+=(cuFloatComplex &a, T &b)
{
    a = cuCadd(a, make_cuFloatComplex(b));
}
#endif
